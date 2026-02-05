import os
from abc import ABC, abstractmethod

import torch
import torchvision.transforms as transforms
from einops import rearrange
from torchvision.datasets.utils import download_url
from typing import Optional, Tuple


class BaseReward(ABC):
    """An base class for reward models. A custom Reward class must implement two functions below."""

    def __init__(self):
        """Define your reward model and image transformations (optional) here."""
        pass

    @abstractmethod
    def __call__(
        self, batch_frames: torch.Tensor, batch_prompt: Optional[list[str]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given batch frames with shape `[B, C, T, H, W]` extracted from a list of videos and a list of prompts
        (optional) correspondingly, return the loss and reward computed by your reward model (reduction by mean).
        """
        pass


class VCDReward(BaseReward):
    """Video Consistency Distance reward for reward-LoRA training.

    This reward follows the paper-style frequency-domain consistency objective:
    lower VCD is better, so reward is defined as `-VCD`, and loss minimizes VCD.
    """

    def __init__(
        self,
        device="cpu",
        dtype=torch.float16,  # kept for API compatibility
        loss_scale: float = 1.0,
        use_amplitude: bool = True,
        use_phase: bool = True,
        amplitude_weight: float = 1.0,
        phase_weight: float = 1.0,
        num_sampled_frames: int = 4,
        random_frame_sampling: bool = True,
        use_temporal_weight: bool = True,
        feature_layers=None,
        feature_resolution: int = 224,
        max_coeffs: int = 16384,
        random_coeff_sampling: bool = True,
        use_pretrained_vgg: bool = True,
        conditioning_source: str = "first_generated_frame",
        detach_conditioning_frame: bool = True,
        assume_neg_one_to_one: bool = False,
        **_: object,
    ):
        from enhancements.video_consistency_distance.loss import (
            VideoConsistencyDistanceLoss,
            VideoConsistencyDistanceLossConfig,
        )

        self.device = torch.device(device)
        self.dtype = dtype
        self.loss_scale = float(loss_scale)
        if self.loss_scale <= 0:
            raise ValueError(f"loss_scale must be > 0, got {self.loss_scale}")

        self.conditioning_source = str(conditioning_source).lower()
        if self.conditioning_source not in {
            "first_generated_frame",
            "provided_first_frame",
        }:
            raise ValueError(
                "conditioning_source must be 'first_generated_frame' or "
                f"'provided_first_frame', got '{self.conditioning_source}'."
            )

        cfg = VideoConsistencyDistanceLossConfig(
            use_amplitude=bool(use_amplitude),
            use_phase=bool(use_phase),
            amplitude_weight=float(amplitude_weight),
            phase_weight=float(phase_weight),
            num_sampled_frames=int(num_sampled_frames),
            random_frame_sampling=bool(random_frame_sampling),
            use_temporal_weight=bool(use_temporal_weight),
            start_step=0,
            end_step=None,
            warmup_steps=0,
            apply_every_n_steps=1,
            feature_layers=tuple(
                feature_layers
                if feature_layers is not None
                else [1, 6, 11, 20, 29]
            ),
            feature_resolution=int(feature_resolution),
            max_coeffs=int(max_coeffs),
            random_coeff_sampling=bool(random_coeff_sampling),
            use_pretrained_vgg=bool(use_pretrained_vgg),
            detach_conditioning_frame=bool(detach_conditioning_frame),
            assume_neg_one_to_one=bool(assume_neg_one_to_one),
        )
        self.vcd = VideoConsistencyDistanceLoss(cfg, device=self.device)

    def __call__(
        self, batch_frames: torch.Tensor, batch_prompt: Optional[list[str]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if batch_frames.dim() != 5:
            raise ValueError(
                f"VCDReward expects [B, C, T, H, W], got {batch_frames.shape}"
            )

        # [B, C, T, H, W] -> [B, T, C, H, W]
        frames_btchw = rearrange(batch_frames, "b c t h w -> b t c h w")
        if self.conditioning_source == "first_generated_frame":
            conditioning_frame = frames_btchw[:, 0, :, :, :]
        else:
            # Placeholder mode for future external conditioning-frame injection.
            conditioning_frame = frames_btchw[:, 0, :, :, :]

        vcd_loss = self.vcd.compute(
            pred_frames=frames_btchw,
            conditioning_frame=conditioning_frame,
            step=0,
        )
        if vcd_loss is None:
            zero = batch_frames.new_zeros(())
            return zero, zero

        loss = vcd_loss * self.loss_scale
        reward = -vcd_loss.detach()
        return loss, reward


class AestheticReward(BaseReward):
    """Aesthetic Predictor [V2](https://github.com/christophschuhmann/improved-aesthetic-predictor)
    and [V2.5](https://github.com/discus0434/aesthetic-predictor-v2-5) reward model.
    """

    def __init__(
        self,
        encoder_path="openai/clip-vit-large-patch14",
        predictor_path=None,
        version="v2",
        device="cpu",
        dtype=torch.float16,
        max_reward=10,
        loss_scale=0.1,
    ):
        from reward.improved_aesthetic_predictor import ImprovedAestheticPredictor
        from reward.siglip_v2_5 import convert_v2_5_from_siglip

        self.encoder_path = encoder_path
        self.predictor_path = predictor_path
        self.version = version
        self.device = device
        self.dtype = dtype
        self.max_reward = max_reward
        self.loss_scale = loss_scale

        if self.version != "v2" and self.version != "v2.5":
            raise ValueError("Only v2 and v2.5 are supported.")
        if self.version == "v2":
            assert "clip-vit-large-patch14" in encoder_path.lower()
            self.model = ImprovedAestheticPredictor(
                encoder_path=self.encoder_path, predictor_path=self.predictor_path
            )
            # https://huggingface.co/openai/clip-vit-large-patch14/blob/main/preprocessor_config.json
            # TODO: [transforms.Resize(224), transforms.CenterCrop(224)] for any aspect ratio.
            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.Normalize(
                        mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711],
                    ),
                ]
            )
        elif self.version == "v2.5":
            assert "siglip-so400m-patch14-384" in encoder_path.lower()
            self.model, _ = convert_v2_5_from_siglip(
                encoder_model_name=self.encoder_path
            )
            # https://huggingface.co/google/siglip-so400m-patch14-384/blob/main/preprocessor_config.json
            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        (384, 384), interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )

        self.model.to(device=self.device, dtype=self.dtype)  # type: ignore
        self.model.requires_grad_(False)

    def __call__(
        self, batch_frames: torch.Tensor, batch_prompt: Optional[list[str]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_frames = rearrange(batch_frames, "b c t h w -> t b c h w")
        batch_loss, batch_reward = 0, 0
        for frames in batch_frames:
            pixel_values = torch.stack([self.transform(frame) for frame in frames])
            pixel_values = pixel_values.to(self.device, dtype=self.dtype)
            if self.version == "v2":
                reward = self.model(pixel_values)
            elif self.version == "v2.5":
                reward = self.model(pixel_values).logits.squeeze()
            # Convert reward to loss in [0, 1].
            if self.max_reward is None:
                loss = (-1 * reward) * self.loss_scale
            else:
                loss = abs(reward - self.max_reward) * self.loss_scale
            batch_loss, batch_reward = (
                batch_loss + loss.mean(),  # type: ignore
                batch_reward + reward.mean(),
            )

        return batch_loss / batch_frames.shape[0], batch_reward / batch_frames.shape[0]  # type: ignore


class HPSReward(BaseReward):
    """[HPS](https://github.com/tgxs002/HPSv2) v2 and v2.1 reward model."""

    def __init__(
        self,
        model_path=None,
        version="v2.0",
        device="cpu",
        dtype=torch.float16,
        max_reward=1,
        loss_scale=1,
    ):
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

        self.model_path = model_path
        self.version = version
        self.device = device
        self.dtype = dtype
        self.max_reward = max_reward
        self.loss_scale = loss_scale

        self.model, _, _ = create_model_and_transforms(
            "ViT-H-14",
            "laion2B-s32B-b79K",
            precision=self.dtype,  # type: ignore
            device=self.device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False,
        )
        self.tokenizer = get_tokenizer("ViT-H-14")

        # https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/blob/main/preprocessor_config.json
        # TODO: [transforms.Resize(224), transforms.CenterCrop(224)] for any aspect ratio.
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

        if version == "v2.0":
            url = "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Third_Party/HPS_v2_compressed.pt"
            filename = "HPS_v2_compressed.pt"
            md5 = "fd9180de357abf01fdb4eaad64631db4"
        elif version == "v2.1":
            url = "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Third_Party/HPS_v2.1_compressed.pt"
            filename = "HPS_v2.1_compressed.pt"
            md5 = "4067542e34ba2553a738c5ac6c1d75c0"
        else:
            raise ValueError("Only v2.0 and v2.1 are supported.")
        if self.model_path is None or not os.path.exists(self.model_path):
            download_url(url, torch.hub.get_dir(), md5=md5)
            model_path = os.path.join(torch.hub.get_dir(), filename)

        state_dict = torch.load(model_path, map_location="cpu")["state_dict"]  # type: ignore
        self.model.load_state_dict(state_dict)
        self.model.to(device=self.device, dtype=self.dtype)
        self.model.requires_grad_(False)
        self.model.eval()

    def __call__(
        self, batch_frames: torch.Tensor, batch_prompt: list[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert batch_frames.shape[0] == len(batch_prompt)
        # Compute batch reward and loss in frame-wise.
        batch_frames = rearrange(batch_frames, "b c t h w -> t b c h w")
        batch_loss, batch_reward = 0, 0
        for frames in batch_frames:
            image_inputs = torch.stack([self.transform(frame) for frame in frames])
            image_inputs = image_inputs.to(device=self.device, dtype=self.dtype)
            text_inputs = self.tokenizer(batch_prompt).to(device=self.device)
            outputs = self.model(image_inputs, text_inputs)

            image_features, text_features = (
                outputs["image_features"],
                outputs["text_features"],
            )
            logits = image_features @ text_features.T
            reward = torch.diagonal(logits)
            # Convert reward to loss in [0, 1].
            if self.max_reward is None:
                loss = (-1 * reward) * self.loss_scale
            else:
                loss = abs(reward - self.max_reward) * self.loss_scale

            batch_loss, batch_reward = (
                batch_loss + loss.mean(),
                batch_reward + reward.mean(),
            )

        return batch_loss / batch_frames.shape[0], batch_reward / batch_frames.shape[0]  # type: ignore


class PickScoreReward(BaseReward):
    """[PickScore](https://github.com/yuvalkirstain/PickScore) reward model."""

    def __init__(
        self,
        model_path="yuvalkirstain/PickScore_v1",
        device="cpu",
        dtype=torch.float16,
        max_reward=1,
        loss_scale=1,
    ):
        from transformers import AutoProcessor, AutoModel

        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.max_reward = max_reward
        self.loss_scale = loss_scale

        # https://huggingface.co/yuvalkirstain/PickScore_v1/blob/main/preprocessor_config.json
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        self.processor = AutoProcessor.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", torch_dtype=self.dtype
        )
        self.model = (
            AutoModel.from_pretrained(model_path, torch_dtype=self.dtype)
            .eval()
            .to(device)
        )
        self.model.requires_grad_(False)
        self.model.eval()

    def __call__(
        self, batch_frames: torch.Tensor, batch_prompt: list[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert batch_frames.shape[0] == len(batch_prompt)
        # Compute batch reward and loss in frame-wise.
        batch_frames = rearrange(batch_frames, "b c t h w -> t b c h w")
        batch_loss, batch_reward = 0, 0
        for frames in batch_frames:
            image_inputs = torch.stack([self.transform(frame) for frame in frames])
            image_inputs = image_inputs.to(device=self.device, dtype=self.dtype)
            text_inputs = self.processor(
                text=batch_prompt,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(self.device)
            image_features = self.model.get_image_features(pixel_values=image_inputs)
            text_features = self.model.get_text_features(**text_inputs)
            image_features = image_features / torch.norm(
                image_features, dim=-1, keepdim=True
            )
            text_features = text_features / torch.norm(
                text_features, dim=-1, keepdim=True
            )

            logits = image_features @ text_features.T
            reward = torch.diagonal(logits)
            # Convert reward to loss in [0, 1].
            if self.max_reward is None:
                loss = (-1 * reward) * self.loss_scale
            else:
                loss = abs(reward - self.max_reward) * self.loss_scale

            batch_loss, batch_reward = (
                batch_loss + loss.mean(),
                batch_reward + reward.mean(),
            )

        return batch_loss / batch_frames.shape[0], batch_reward / batch_frames.shape[0]  # type: ignore


class MPSReward(BaseReward):
    """[MPS](https://github.com/Kwai-Kolors/MPS) reward model."""

    def __init__(
        self,
        model_path=None,
        device="cpu",
        dtype=torch.float16,
        max_reward=1,
        loss_scale=1,
    ):
        from transformers import AutoTokenizer, AutoConfig
        from reward.clip_model import CLIPModel

        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.condition = "light, color, clarity, tone, style, ambiance, artistry, shape, face, hair, hands, limbs, structure, instance, texture, quantity, attributes, position, number, location, word, things."
        self.max_reward = max_reward
        self.loss_scale = loss_scale

        processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        # https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/blob/main/preprocessor_config.json
        # TODO: [transforms.Resize(224), transforms.CenterCrop(224)] for any aspect ratio.
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

        # We convert the original [ckpt](http://drive.google.com/file/d/17qrK_aJkVNM75ZEvMEePpLj6L867MLkN/view?usp=sharing)
        # (contains the entire model) to a `state_dict`.
        url = "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Third_Party/MPS_overall.pth"
        filename = "MPS_overall.pth"
        md5 = "1491cbbbd20565747fe07e7572e2ac56"
        if self.model_path is None or not os.path.exists(self.model_path):
            download_url(url, torch.hub.get_dir(), md5=md5)
            model_path = os.path.join(torch.hub.get_dir(), filename)

        self.tokenizer = AutoTokenizer.from_pretrained(
            processor_name_or_path, trust_remote_code=True
        )
        config = AutoConfig.from_pretrained(processor_name_or_path)
        self.model = CLIPModel(config)
        state_dict = torch.load(model_path, map_location="cpu")  # type: ignore
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(device=self.device, dtype=self.dtype)
        self.model.requires_grad_(False)
        self.model.eval()

    def _tokenize(self, caption):
        input_ids = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        return input_ids

    def __call__(
        self,
        batch_frames: torch.Tensor,
        batch_prompt: list[str],
        batch_condition: Optional[list[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if batch_condition is None:
            batch_condition = [self.condition] * len(batch_prompt)
        batch_frames = rearrange(batch_frames, "b c t h w -> t b c h w")
        batch_loss, batch_reward = 0, 0
        for frames in batch_frames:
            image_inputs = torch.stack([self.transform(frame) for frame in frames])
            image_inputs = image_inputs.to(device=self.device, dtype=self.dtype)
            text_inputs = self._tokenize(batch_prompt).to(self.device)
            condition_inputs = self._tokenize(batch_condition).to(device=self.device)
            text_features, image_features = self.model(
                text_inputs, image_inputs, condition_inputs
            )

            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            # reward = self.model.logit_scale.exp() * torch.diag(torch.einsum('bd,cd->bc', text_features, image_features))
            logits = image_features @ text_features.T
            reward = torch.diagonal(logits)
            # Convert reward to loss in [0, 1].
            if self.max_reward is None:
                loss = (-1 * reward) * self.loss_scale
            else:
                loss = abs(reward - self.max_reward) * self.loss_scale

            batch_loss, batch_reward = (
                batch_loss + loss.mean(),
                batch_reward + reward.mean(),
            )

        return batch_loss / batch_frames.shape[0], batch_reward / batch_frames.shape[0]  # type: ignore


if __name__ == "__main__":
    import numpy as np
    from decord import VideoReader

    video_path_list = ["your_video_path_1.mp4", "your_video_path_2.mp4"]
    prompt_list = ["your_prompt_1", "your_prompt_2"]
    num_sampled_frames = 8

    to_tensor = transforms.ToTensor()

    sampled_frames_list = []
    for video_path in video_path_list:
        vr = VideoReader(video_path)
        sampled_frame_indices = np.linspace(
            0, len(vr), num_sampled_frames, endpoint=False, dtype=int
        )
        sampled_frames = vr.get_batch(sampled_frame_indices).asnumpy()
        sampled_frames = torch.stack([to_tensor(frame) for frame in sampled_frames])
        sampled_frames_list.append(sampled_frames)
    sampled_frames = torch.stack(sampled_frames_list)
    sampled_frames = rearrange(sampled_frames, "b t c h w -> b c t h w")

    aesthetic_reward_v2 = AestheticReward(device="cuda", dtype=torch.bfloat16)
    print(f"aesthetic_reward_v2: {aesthetic_reward_v2(sampled_frames)}")

    aesthetic_reward_v2_5 = AestheticReward(
        encoder_path="google/siglip-so400m-patch14-384",
        version="v2.5",
        device="cuda",
        dtype=torch.bfloat16,
    )
    print(f"aesthetic_reward_v2_5: {aesthetic_reward_v2_5(sampled_frames)}")

    hps_reward_v2 = HPSReward(device="cuda", dtype=torch.bfloat16)
    print(f"hps_reward_v2: {hps_reward_v2(sampled_frames, prompt_list)}")

    hps_reward_v2_1 = HPSReward(version="v2.1", device="cuda", dtype=torch.bfloat16)
    print(f"hps_reward_v2_1: {hps_reward_v2_1(sampled_frames, prompt_list)}")

    pick_score = PickScoreReward(device="cuda", dtype=torch.bfloat16)
    print(f"pick_score_reward: {pick_score(sampled_frames, prompt_list)}")

    mps_score = MPSReward(device="cuda", dtype=torch.bfloat16)
    print(f"mps_reward: {mps_score(sampled_frames, prompt_list)}")
