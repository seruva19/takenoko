# Slider Training Core - Concept editing training implementation
# Implements concept slider training with dual-polarity LoRA approach

from typing import Optional, Dict, Any, List, Union, Tuple
import os
import torch
import torch.nn.functional as F
from torch import Tensor
from safetensors.torch import load_file, save_file

from utils.train_utils import get_torch_dtype
from modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler

import logging
from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


class PromptEmbeds:
    # text_embeds: torch.Tensor
    # pooled_embeds: Union[torch.Tensor, None]
    # attention_mask: Union[torch.Tensor, List[torch.Tensor], None]

    def __init__(
        self,
        args: Union[Tuple[torch.Tensor], List[torch.Tensor], torch.Tensor],
        attention_mask=None,
    ) -> None:
        if isinstance(args, list) or isinstance(args, tuple):
            # xl
            self.text_embeds = args[0]
            self.pooled_embeds = args[1]
        else:
            # sdv1.x, sdv2.x
            self.text_embeds = args
            self.pooled_embeds = None

        self.attention_mask = attention_mask

    def to(self, *args, **kwargs):
        if isinstance(self.text_embeds, list) or isinstance(self.text_embeds, tuple):
            self.text_embeds = [t.to(*args, **kwargs) for t in self.text_embeds]
        else:
            self.text_embeds = self.text_embeds.to(*args, **kwargs)
        if self.pooled_embeds is not None:
            self.pooled_embeds = self.pooled_embeds.to(*args, **kwargs)
        if self.attention_mask is not None:
            if isinstance(self.attention_mask, list) or isinstance(
                self.attention_mask, tuple
            ):
                self.attention_mask = [
                    t.to(*args, **kwargs) for t in self.attention_mask
                ]
            else:
                self.attention_mask = self.attention_mask.to(*args, **kwargs)
        return self

    def detach(self):
        new_embeds = self.clone()
        if isinstance(new_embeds.text_embeds, list) or isinstance(
            new_embeds.text_embeds, tuple
        ):
            new_embeds.text_embeds = [t.detach() for t in new_embeds.text_embeds]
        else:
            new_embeds.text_embeds = new_embeds.text_embeds.detach()
        if new_embeds.pooled_embeds is not None:
            new_embeds.pooled_embeds = new_embeds.pooled_embeds.detach()
        if new_embeds.attention_mask is not None:
            if isinstance(new_embeds.attention_mask, list) or isinstance(
                new_embeds.attention_mask, tuple
            ):
                new_embeds.attention_mask = [
                    t.detach() for t in new_embeds.attention_mask
                ]
            else:
                new_embeds.attention_mask = new_embeds.attention_mask.detach()
        return new_embeds

    def clone(self):
        if isinstance(self.text_embeds, list) or isinstance(self.text_embeds, tuple):
            cloned_text_embeds = [t.clone() for t in self.text_embeds]
        else:
            cloned_text_embeds = self.text_embeds.clone()
        if self.pooled_embeds is not None:
            prompt_embeds = PromptEmbeds(
                [cloned_text_embeds, self.pooled_embeds.clone()]
            )
        else:
            prompt_embeds = PromptEmbeds(cloned_text_embeds)

        if self.attention_mask is not None:
            if isinstance(self.attention_mask, list) or isinstance(
                self.attention_mask, tuple
            ):
                prompt_embeds.attention_mask = [t.clone() for t in self.attention_mask]
            else:
                prompt_embeds.attention_mask = self.attention_mask.clone()
        return prompt_embeds

    def expand_to_batch(self, batch_size):
        pe = self.clone()
        if isinstance(pe.text_embeds, list) or isinstance(pe.text_embeds, tuple):
            current_batch_size = pe.text_embeds[0].shape[0]
        else:
            current_batch_size = pe.text_embeds.shape[0]
        if current_batch_size == batch_size:
            return pe
        if current_batch_size != 1:
            raise Exception("Can only expand batch size for batch size 1")
        if isinstance(pe.text_embeds, list) or isinstance(pe.text_embeds, tuple):
            pe.text_embeds = [t.expand(batch_size, -1) for t in pe.text_embeds]
        else:
            pe.text_embeds = pe.text_embeds.expand(batch_size, -1)
        if pe.pooled_embeds is not None:
            pe.pooled_embeds = pe.pooled_embeds.expand(batch_size, -1)
        if pe.attention_mask is not None:
            if isinstance(pe.attention_mask, list) or isinstance(
                pe.attention_mask, tuple
            ):
                pe.attention_mask = [
                    t.expand(batch_size, -1) for t in pe.attention_mask
                ]
            else:
                pe.attention_mask = pe.attention_mask.expand(batch_size, -1)
        return pe

    def save(self, path: str):
        """
        Save the prompt embeds to a file.
        :param path: The path to save the prompt embeds.
        """
        pe = self.clone()
        state_dict = {}
        if isinstance(pe.text_embeds, list) or isinstance(pe.text_embeds, tuple):
            for i, text_embed in enumerate(pe.text_embeds):
                state_dict[f"text_embed_{i}"] = text_embed.cpu()
        else:
            state_dict["text_embed"] = pe.text_embeds.cpu()

        if pe.pooled_embeds is not None:
            state_dict["pooled_embed"] = pe.pooled_embeds.cpu()
        if pe.attention_mask is not None:
            if isinstance(pe.attention_mask, list) or isinstance(
                pe.attention_mask, tuple
            ):
                for i, attn in enumerate(pe.attention_mask):
                    state_dict[f"attention_mask_{i}"] = attn.cpu()
            else:
                state_dict["attention_mask"] = pe.attention_mask.cpu()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_file(state_dict, path)

    @classmethod
    def load(cls, path: str) -> "PromptEmbeds":
        """
        Load the prompt embeds from a file.
        :param path: The path to load the prompt embeds from.
        :return: An instance of PromptEmbeds.
        """
        state_dict = load_file(path, device="cpu")
        text_embeds = []
        pooled_embeds = None
        attention_mask = []
        for key in sorted(state_dict.keys()):
            if key.startswith("text_embed_"):
                text_embeds.append(state_dict[key])
            elif key == "text_embed":
                text_embeds.append(state_dict[key])
            elif key == "pooled_embed":
                pooled_embeds = state_dict[key]
            elif key.startswith("attention_mask_"):
                attention_mask.append(state_dict[key])
            elif key == "attention_mask":
                attention_mask.append(state_dict[key])
        pe = cls(None)
        pe.text_embeds = text_embeds
        if len(text_embeds) == 1:
            pe.text_embeds = text_embeds[0]
        if pooled_embeds is not None:
            pe.pooled_embeds = pooled_embeds
        if len(attention_mask) > 0:
            if len(attention_mask) == 1:
                pe.attention_mask = attention_mask[0]
            else:
                pe.attention_mask = attention_mask
        return pe


def concat_prompt_embeds(prompt_embeds: list["PromptEmbeds"]):
    # --- pad text_embeds ---
    if isinstance(prompt_embeds[0].text_embeds, (list, tuple)):
        embed_list = []
        for i in range(len(prompt_embeds[0].text_embeds)):
            max_len = max(p.text_embeds[i].shape[1] for p in prompt_embeds)
            padded = []
            for p in prompt_embeds:
                t = p.text_embeds[i]
                if t.shape[1] < max_len:
                    pad = torch.zeros(
                        (t.shape[0], max_len - t.shape[1], *t.shape[2:]),
                        dtype=t.dtype,
                        device=t.device,
                    )
                    t = torch.cat([t, pad], dim=1)
                padded.append(t)
            embed_list.append(torch.cat(padded, dim=0))
        text_embeds = embed_list
    else:
        max_len = max(p.text_embeds.shape[1] for p in prompt_embeds)
        padded = []
        for p in prompt_embeds:
            t = p.text_embeds
            if t.shape[1] < max_len:
                pad = torch.zeros(
                    (t.shape[0], max_len - t.shape[1], *t.shape[2:]),
                    dtype=t.dtype,
                    device=t.device,
                )
                t = torch.cat([t, pad], dim=1)
            padded.append(t)
        text_embeds = torch.cat(padded, dim=0)

    # --- pooled embeds ---
    pooled_embeds = None
    if prompt_embeds[0].pooled_embeds is not None:
        pooled_embeds = torch.cat([p.pooled_embeds for p in prompt_embeds], dim=0)

    # --- attention mask ---
    attention_mask = None
    if prompt_embeds[0].attention_mask is not None:
        max_len = max(p.attention_mask.shape[1] for p in prompt_embeds)
        padded = []
        for p in prompt_embeds:
            m = p.attention_mask
            if m.shape[1] < max_len:
                pad = torch.zeros(
                    (m.shape[0], max_len - m.shape[1]),
                    dtype=m.dtype,
                    device=m.device,
                )
                m = torch.cat([m, pad], dim=1)
            padded.append(m)
        attention_mask = torch.cat(padded, dim=0)

    # wrap back into PromptEmbeds
    pe = PromptEmbeds([text_embeds, pooled_embeds])
    pe.attention_mask = attention_mask
    return pe


def norm_like_tensor(tensor: Tensor, target: Tensor) -> Tensor:
    """
    Normalize the tensor to have the same mean and std as the target tensor.

    Args:
        tensor: Input tensor to normalize
        target: Target tensor providing mean/std reference

    Returns:
        Normalized tensor with target's statistics
    """
    tensor_mean = tensor.mean()
    tensor_std = tensor.std()
    target_mean = target.mean()
    target_std = target.std()

    normalized_tensor = (tensor - tensor_mean) / (
        tensor_std + 1e-8
    ) * target_std + target_mean

    return normalized_tensor


class SliderTrainingCore:
    """
    Core slider training implementation that computes guided loss for concept editing.

    This class handles the dual-polarity training approach where a LoRA is trained
    with both positive and negative multipliers to enhance or suppress concepts.
    """

    def __init__(
        self,
        guidance_strength: float = 3.0,
        anchor_strength: float = 1.0,
        guidance_scale: float = 1.0,
        guidance_embedding_scale: float = 1.0,
        target_guidance_scale: float = 1.0,
        positive_prompt: str = "",
        negative_prompt: str = "",
        target_class: str = "",
        anchor_class: Optional[str] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        t5_device: str = "cpu",
        cache_on_init: bool = True,
    ):
        """
        Initialize Slider Training Core.

        Args:
            guidance_strength: Strength of concept guidance (default: 3.0)
            anchor_strength: Strength of anchor class preservation (default: 1.0)
            guidance_scale: Base classifier-free guidance scale (default: 1.0)
            guidance_embedding_scale: Embedding-level guidance scaling (default: 1.0)
            target_guidance_scale: Separate scale for target predictions (default: 1.0)
            positive_prompt: Prompt representing positive concept direction
            negative_prompt: Prompt representing negative concept direction
            target_class: Main class/concept being edited
            anchor_class: Optional anchor class for preservation
            device: Torch device for computations
            dtype: Data type for computations
            t5_device: Device for T5 encoder ("cpu" or "cuda")
            cache_on_init: Whether to cache embeddings on initialization
        """
        self.guidance_strength = guidance_strength
        self.anchor_strength = anchor_strength
        self.guidance_scale = guidance_scale
        self.guidance_embedding_scale = guidance_embedding_scale
        self.target_guidance_scale = target_guidance_scale
        self.positive_prompt = positive_prompt
        self.negative_prompt = negative_prompt
        self.target_class = target_class
        self.anchor_class = anchor_class
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = dtype or torch.bfloat16

        # T5 configuration (follows Takenoko's pattern)
        self.t5_device = torch.device(t5_device)
        self.cache_on_init = cache_on_init
        self.t5_encoder = None

        # Cached embeddings - will be computed once during setup
        self.positive_prompt_embeds: Optional[PromptEmbeds] = None
        self.negative_prompt_embeds: Optional[PromptEmbeds] = None
        self.target_class_embeds: Optional[PromptEmbeds] = None
        self.anchor_class_embeds: Optional[PromptEmbeds] = None

        self.is_initialized = False

        # For now, initialize with dummy embeddings that are consistent
        self._initialize_dummy_embeddings()

        logger.info(f"✅ Slider Training Core initialized:")
        logger.info(f"   • Guidance strength: {guidance_strength}")
        logger.info(f"   • Anchor strength: {anchor_strength}")
        logger.info(f"   • Guidance scale: {guidance_scale}")
        logger.info(f"   • Guidance embedding scale: {guidance_embedding_scale}")
        logger.info(f"   • Target guidance scale: {target_guidance_scale}")
        logger.info(f"   • Target class: '{target_class}'")
        logger.info(f"   • Anchor class: '{anchor_class}'")

    def _initialize_dummy_embeddings(self) -> None:
        """
        Initialize consistent dummy embeddings for testing.
        These embeddings will be the same across all calls, unlike random generation.
        """
        # Create consistent dummy embeddings using fixed seeds
        torch.manual_seed(42)  # Fixed seed for consistency

        # Standard T5 dimensions
        text_dim = 2048
        seq_len = 77

        # Create consistent dummy embeddings for each prompt
        self.positive_prompt_embeds = PromptEmbeds(
            [
                torch.randn(1, seq_len, text_dim, device=self.device, dtype=self.dtype),
                torch.randn(1, text_dim, device=self.device, dtype=self.dtype),
            ]
        )

        # Use different seeds for different concepts to ensure they're distinct
        torch.manual_seed(43)
        self.target_class_embeds = PromptEmbeds(
            [
                torch.randn(1, seq_len, text_dim, device=self.device, dtype=self.dtype),
                torch.randn(1, text_dim, device=self.device, dtype=self.dtype),
            ]
        )

        torch.manual_seed(44)
        self.negative_prompt_embeds = PromptEmbeds(
            [
                torch.randn(1, seq_len, text_dim, device=self.device, dtype=self.dtype),
                torch.randn(1, text_dim, device=self.device, dtype=self.dtype),
            ]
        )

        if self.anchor_class:
            torch.manual_seed(45)
            self.anchor_class_embeds = PromptEmbeds(
                [
                    torch.randn(
                        1, seq_len, text_dim, device=self.device, dtype=self.dtype
                    ),
                    torch.randn(1, text_dim, device=self.device, dtype=self.dtype),
                ]
            )

        # Reset random seed
        torch.manual_seed(torch.initial_seed())

        logger.info(
            "🔧 Initialized consistent dummy embeddings (deterministic for concept directions)"
        )

    def _load_t5_encoder(self, args=None) -> Optional[Any]:
        """
        Load T5 encoder using Takenoko's pattern.

        Args:
            args: Training arguments containing T5 configuration

        Returns:
            T5 encoder model or None if loading fails
        """
        try:
            # Import T5 encoder from WAN modules (same as Takenoko)
            from wan.modules.t5 import T5EncoderModel
            from common.model_downloader import download_model_if_needed

            # Get T5 configuration from args (same as Takenoko)
            if not args:
                logger.warning("No args provided for T5 loading")
                return None

            t5_path = getattr(args, "t5", None)
            if not t5_path:
                logger.warning("No T5 path found in args.t5")
                return None

            fp8_t5 = getattr(args, "fp8_t5", False)

            # Get T5 config from config object (if available)
            config = getattr(args, "config", None) if hasattr(args, "config") else None
            text_len = getattr(config, "text_len", 512) if config else 512
            t5_dtype = (
                getattr(config, "t5_dtype", torch.bfloat16)
                if config
                else torch.bfloat16
            )

            # Download T5 model if it's a URL (same as Takenoko)
            if t5_path.startswith(("http://", "https://")):
                logger.info(f"Detected URL for T5 model, downloading: {t5_path}")
                cache_dir = getattr(args, "model_cache_dir", None)
                t5_path = download_model_if_needed(t5_path, cache_dir=cache_dir)
                logger.info(f"Downloaded T5 model to: {t5_path}")

            logger.info(f"Loading T5 encoder from: {t5_path}")
            logger.info(f"T5 device: {self.t5_device}")

            # Load T5 encoder using Takenoko's exact pattern
            t5_encoder = T5EncoderModel(
                text_len=text_len,
                dtype=t5_dtype,
                device=self.t5_device,
                weight_path=t5_path,
                fp8=fp8_t5,
            )

            logger.info(f"✅ T5 encoder loaded successfully on {self.t5_device}")
            return t5_encoder

        except ImportError as e:
            logger.warning(f"T5EncoderModel not available: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to load T5 encoder: {e}")
            return None

    def _predict_noise(
        self,
        transformer: torch.nn.Module,
        latents: torch.Tensor,
        conditional_embeddings: PromptEmbeds,
        timestep: torch.Tensor,
        guidance_scale: float = 1.0,
        guidance_embedding_scale: float = 1.0,
        batch: Dict[str, Any] = None,
    ) -> torch.Tensor:
        """
        Predict noise using Takenoko's transformer for slider training.

        This provides a consistent interface for noise prediction during slider training.

        Args:
            transformer: The transformer model
            latents: Input latent tensors
            conditional_embeddings: Text conditioning embeddings
            timestep: Timestep values
            guidance_scale: Classifier-free guidance scale (default: 1.0 for slider training)
            guidance_embedding_scale: Embedding guidance scale (default: 1.0)
            batch: Additional batch data

        Returns:
            Noise prediction tensor
        """
        # Apply guidance embedding scaling to conditioning
        encoder_hidden_states = conditional_embeddings.text_embeds
        if guidance_embedding_scale != 1.0:
            encoder_hidden_states = encoder_hidden_states * guidance_embedding_scale

        # Get model prediction
        pred = transformer(
            latents,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]

        # Apply guidance scaling to output prediction
        if guidance_scale != 1.0:
            pred = pred * guidance_scale

        return pred

    def initialize_embeddings(
        self,
        sd_model=None,
        text_encoder=None,
        cache_text_embeddings: bool = True,
        args=None,
    ) -> None:
        """
        Initialize and cache text embeddings for slider training.

        Args:
            sd_model: Optional stable diffusion model with text encoder
            text_encoder: Optional T5 text encoder for real embedding computation
            cache_text_embeddings: Whether to cache embeddings
            args: Training arguments for T5 loading
        """
        if self.is_initialized:
            return

        # Try to load T5 encoder if not provided (follows Takenoko's pattern)
        if text_encoder is None:
            self.t5_encoder = self._load_t5_encoder(args)
            if self.t5_encoder is not None:
                text_encoder = self.t5_encoder

        self.text_encoder = text_encoder

        # If we have a text encoder and caching is enabled, pre-compute embeddings
        if text_encoder is not None and (cache_text_embeddings or self.cache_on_init):
            try:
                logger.info("Pre-computing real T5 embeddings for slider training...")

                # Pre-compute embeddings for all prompts
                self.positive_prompt_embeds = self._encode_prompt_on_demand(
                    self.positive_prompt, batch_size=1, text_encoder=text_encoder
                )
                self.target_class_embeds = self._encode_prompt_on_demand(
                    self.target_class, batch_size=1, text_encoder=text_encoder
                )
                self.negative_prompt_embeds = self._encode_prompt_on_demand(
                    self.negative_prompt, batch_size=1, text_encoder=text_encoder
                )

                if self.anchor_class:
                    self.anchor_class_embeds = self._encode_prompt_on_demand(
                        self.anchor_class, batch_size=1, text_encoder=text_encoder
                    )

                logger.info(
                    "✅ Slider training initialized with real T5 text embeddings"
                )

                # Cleanup T5 encoder to free memory (like Takenoko does)
                if self.t5_encoder is not None:
                    del self.t5_encoder
                    self.t5_encoder = None

            except Exception as e:
                logger.warning(f"Failed to pre-compute T5 embeddings: {e}")
                logger.info("Will use consistent dummy embeddings as fallback")
                self._initialize_dummy_embeddings()
        else:
            # Use consistent dummy embeddings as fallback
            logger.info("No T5 encoder available, using consistent dummy embeddings")
            self._initialize_dummy_embeddings()

        self.is_initialized = True

    def _encode_prompt_on_demand(
        self, prompt: str, batch_size: int = 1, text_encoder=None
    ) -> PromptEmbeds:
        """
        Encode prompts on-demand using available text encoder or consistent fallback.

        Args:
            prompt: Text prompt to encode
            batch_size: Batch size for encoding
            text_encoder: Optional T5 encoder model for real encoding

        Returns:
            Encoded prompt embeddings
        """
        # Validate inputs
        if not isinstance(prompt, str):
            raise TypeError(f"Prompt must be a string, got {type(prompt)}")

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty or whitespace-only")

        if batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {batch_size}")

        if len(prompt) > 2048:  # Reasonable limit for T5
            logger.warning(
                f"Prompt is very long ({len(prompt)} chars), may be truncated by T5"
            )
        # Try to use real text encoder if available (using Takenoko's pattern)
        if text_encoder is not None:
            try:
                # Use Takenoko's T5 calling pattern with autocast and no_grad
                device = self.t5_device
                t5_dtype = getattr(text_encoder, "dtype", torch.bfloat16)

                with (
                    torch.autocast(device_type=device.type, dtype=t5_dtype),
                    torch.no_grad(),
                ):
                    # Call T5 encoder exactly like Takenoko: text_encoder([prompt], device)
                    encoded_outputs = text_encoder([prompt], device)

                if encoded_outputs and len(encoded_outputs) > 0:
                    # encoded_outputs is a list of tensors from T5 encoder
                    text_embeds = encoded_outputs[0]  # Shape: [seq_len, hidden_dim]

                    # Add batch dimension: [1, seq_len, hidden_dim]
                    if text_embeds.dim() == 2:
                        text_embeds = text_embeds.unsqueeze(0)

                    # Create pooled embeddings by taking the mean over sequence dimension
                    pooled_embeds = text_embeds.mean(dim=1)  # [batch, hidden_dim]

                    # Move to target device if T5 is on different device
                    if device != self.device:
                        text_embeds = text_embeds.to(self.device, dtype=self.dtype)
                        pooled_embeds = pooled_embeds.to(self.device, dtype=self.dtype)

                    # Replicate for batch size if needed
                    if batch_size > 1:
                        text_embeds = text_embeds.repeat(batch_size, 1, 1)
                        pooled_embeds = pooled_embeds.repeat(batch_size, 1)

                    logger.debug(
                        f"✅ Real T5 encoded prompt: '{prompt[:30]}...'"
                        + f" -> text_embeds: {text_embeds.shape}, pooled: {pooled_embeds.shape}"
                    )
                    return PromptEmbeds([text_embeds, pooled_embeds])

            except Exception as e:
                logger.warning(
                    f"Failed to encode prompt '{prompt}' with T5 encoder: {e}"
                )
                # Fall through to consistent dummy embeddings

        # Fallback: Use consistent dummy embeddings (same across calls)
        # First check if we have cached embeddings for this prompt
        if hasattr(self, "positive_prompt_embeds") and prompt == self.positive_prompt:
            return self._replicate_for_batch(self.positive_prompt_embeds, batch_size)
        elif hasattr(self, "negative_prompt_embeds") and prompt == self.negative_prompt:
            return self._replicate_for_batch(self.negative_prompt_embeds, batch_size)
        elif hasattr(self, "target_class_embeds") and prompt == self.target_class:
            return self._replicate_for_batch(self.target_class_embeds, batch_size)
        elif (
            hasattr(self, "anchor_class_embeds")
            and self.anchor_class
            and prompt == self.anchor_class
        ):
            return self._replicate_for_batch(self.anchor_class_embeds, batch_size)

        # Generate consistent dummy embeddings for unknown prompts
        # Use hash of prompt as seed for consistency
        prompt_hash = hash(prompt) % (2**31)  # Ensure positive
        torch.manual_seed(prompt_hash)

        dummy_text_embed = torch.randn(
            batch_size, 77, 2048, device=self.device, dtype=self.dtype
        )
        dummy_pooled_embed = torch.randn(
            batch_size, 2048, device=self.device, dtype=self.dtype
        )

        # Reset seed
        torch.manual_seed(torch.initial_seed())

        logger.debug(
            f"🔧 Generated consistent dummy embeddings for: '{prompt[:30]}...'"
        )
        return PromptEmbeds([dummy_text_embed, dummy_pooled_embed])

    def _replicate_for_batch(
        self, prompt_embeds: PromptEmbeds, batch_size: int
    ) -> PromptEmbeds:
        """
        Replicate PromptEmbeds for a specific batch size.

        Args:
            prompt_embeds: Source PromptEmbeds to replicate
            batch_size: Target batch size

        Returns:
            PromptEmbeds replicated for the target batch size
        """
        if batch_size <= 1:
            return prompt_embeds

        # Replicate text embeddings
        text_embeds = prompt_embeds.text_embeds
        if text_embeds.shape[0] < batch_size:
            text_embeds = text_embeds.repeat(batch_size, 1, 1)

        # Replicate pooled embeddings if they exist
        pooled_embeds = None
        if prompt_embeds.pooled_embeds is not None:
            pooled_embeds = prompt_embeds.pooled_embeds
            if pooled_embeds.shape[0] < batch_size:
                pooled_embeds = pooled_embeds.repeat(batch_size, 1)

        return PromptEmbeds([text_embeds, pooled_embeds], prompt_embeds.attention_mask)

    def compute_guided_loss(
        self,
        transformer: torch.nn.Module,
        network,
        noisy_latents: Tensor,
        timesteps: Tensor,
        batch: Dict[str, Any],
        noise: Tensor,
        noise_scheduler,
        args,
        accelerator,
        **kwargs,
    ) -> Tensor:
        """
        Compute the guided loss for slider training.

        This implements the core slider training algorithm:
        1. Disable network and get baseline predictions without LoRA
        2. Calculate concept direction targets with guidance scaling
        3. Train with +1.0 multiplier on enhancement/suppression targets
        4. Train with -1.0 multiplier on opposite targets
        5. Combine and return total loss

        Args:
            transformer: The transformer model
            network: The LoRA network being trained
            noisy_latents: Noisy input latents
            timesteps: Timestep values
            batch: Training batch data
            noise: Noise tensor
            noise_scheduler: Noise scheduler (unused but kept for compatibility)
            args: Training arguments
            accelerator: Accelerator instance

        Returns:
            Combined guided loss tensor that requires gradients
        """
        if not self.is_initialized:
            raise RuntimeError(
                "Slider training core not initialized. Call initialize_embeddings() first."
            )

        # Validate runtime inputs
        if noisy_latents is None or noisy_latents.numel() == 0:
            raise ValueError("noisy_latents cannot be None or empty")

        if timesteps is None or timesteps.numel() == 0:
            raise ValueError("timesteps cannot be None or empty")

        if noisy_latents.shape[0] != timesteps.shape[0]:
            raise ValueError(
                f"Batch size mismatch: latents {noisy_latents.shape[0]} vs timesteps {timesteps.shape[0]}"
            )

        if not hasattr(network, "set_multiplier"):
            raise TypeError(
                "Network must support set_multiplier() method for slider training"
            )

        if transformer is None:
            raise ValueError("Transformer model cannot be None")

        # Store original training states
        was_unet_training = transformer.training
        original_multiplier = network.multiplier

        # Disable network for baseline predictions using Takenoko's approach
        network.set_multiplier(0.0)

        # === BASELINE PREDICTIONS WITHOUT LORA (no_grad) ===
        with torch.no_grad():
            transformer.eval()
            noisy_latents = noisy_latents.to(self.device, dtype=self.dtype).detach()
            batch_size = noisy_latents.shape[0]

            # Prepare embeddings (use cached embeddings)
            positive_embeds = concat_prompt_embeds(
                [self.positive_prompt_embeds] * batch_size
            ).to(self.device, dtype=self.dtype)

            target_class_embeds = concat_prompt_embeds(
                [self.target_class_embeds] * batch_size
            ).to(self.device, dtype=self.dtype)

            negative_embeds = concat_prompt_embeds(
                [self.negative_prompt_embeds] * batch_size
            ).to(self.device, dtype=self.dtype)

            anchor_embeds = None
            if self.anchor_class_embeds is not None:
                anchor_embeds = concat_prompt_embeds(
                    [self.anchor_class_embeds] * batch_size
                ).to(self.device, dtype=self.dtype)

            # Combine embeddings for batch processing
            if anchor_embeds is not None:
                combo_embeds = concat_prompt_embeds(
                    [
                        positive_embeds,
                        target_class_embeds,
                        negative_embeds,
                        anchor_embeds,
                    ]
                )
                num_embeds = 4
            else:
                combo_embeds = concat_prompt_embeds(
                    [positive_embeds, target_class_embeds, negative_embeds]
                )
                num_embeds = 3

            # Get baseline predictions in one batch
            combo_pred = self._predict_noise(
                transformer=transformer,
                latents=torch.cat([noisy_latents] * num_embeds, dim=0),
                conditional_embeddings=combo_embeds,
                timestep=torch.cat([timesteps] * num_embeds, dim=0),
                guidance_scale=self.guidance_scale,  # use configurable guidance scale for baseline
                guidance_embedding_scale=self.guidance_embedding_scale,  # use configurable embedding scale
                batch=batch,
            )

            # Split predictions
            if anchor_embeds is not None:
                positive_pred, neutral_pred, negative_pred, anchor_target = (
                    combo_pred.chunk(4, dim=0)
                )
            else:
                positive_pred, neutral_pred, negative_pred = combo_pred.chunk(3, dim=0)
                anchor_target = None

            # Calculate targets using concept direction calculation
            guidance_scale = self.guidance_strength

            # Concept direction computation
            positive = (positive_pred - neutral_pred) - (negative_pred - neutral_pred)
            negative = (negative_pred - neutral_pred) - (positive_pred - neutral_pred)

            enhance_positive_target = neutral_pred + guidance_scale * positive
            enhance_negative_target = neutral_pred + guidance_scale * negative
            erase_negative_target = neutral_pred - guidance_scale * negative
            erase_positive_target = neutral_pred - guidance_scale * positive

            # Normalize targets to match neutral prediction statistics
            enhance_positive_target = norm_like_tensor(
                enhance_positive_target, neutral_pred
            )
            enhance_negative_target = norm_like_tensor(
                enhance_negative_target, neutral_pred
            )
            erase_negative_target = norm_like_tensor(
                erase_negative_target, neutral_pred
            )
            erase_positive_target = norm_like_tensor(
                erase_positive_target, neutral_pred
            )

        # Restore transformer training state
        if was_unet_training:
            transformer.train()

        # Restore network multiplier
        network.set_multiplier(original_multiplier)

        # Prepare embeddings for guided training
        if anchor_embeds is not None:
            embeds = concat_prompt_embeds([target_class_embeds, anchor_embeds])
            noisy_latents_combined = torch.cat([noisy_latents, noisy_latents], dim=0)
            timesteps_combined = torch.cat([timesteps, timesteps], dim=0)
        else:
            embeds = target_class_embeds
            noisy_latents_combined = noisy_latents
            timesteps_combined = timesteps

        # === POSITIVE MULTIPLIER TRAINING ===
        network.set_multiplier(1.0)

        pred_positive = self._predict_noise(
            transformer=transformer,
            latents=noisy_latents_combined,
            conditional_embeddings=embeds,
            timestep=timesteps_combined,
            guidance_scale=self.target_guidance_scale,  # use configurable target guidance scale
            guidance_embedding_scale=self.guidance_embedding_scale,  # use configurable embedding scale
            batch=batch,
        )

        if anchor_embeds is not None:
            class_pred_pos, anchor_pred_pos = pred_positive.chunk(2, dim=0)
        else:
            class_pred_pos = pred_positive
            anchor_pred_pos = None

        # Compute positive losses
        enhance_loss = F.mse_loss(class_pred_pos, enhance_positive_target)
        erase_loss = F.mse_loss(class_pred_pos, erase_negative_target)

        if anchor_target is None:
            anchor_loss = torch.zeros_like(erase_loss)
        else:
            anchor_loss = F.mse_loss(anchor_pred_pos, anchor_target)

        anchor_loss = anchor_loss * self.anchor_strength
        total_pos_loss = (enhance_loss + erase_loss + anchor_loss) / 3.0

        # Send backward now because gradient checkpointing needs network polarity intact
        total_pos_loss.backward()
        total_pos_loss = total_pos_loss.detach()

        # === NEGATIVE MULTIPLIER TRAINING ===
        network.set_multiplier(-1.0)

        pred_negative = self._predict_noise(
            transformer=transformer,
            latents=noisy_latents_combined,
            conditional_embeddings=embeds,
            timestep=timesteps_combined,
            guidance_scale=self.target_guidance_scale,  # use configurable target guidance scale
            guidance_embedding_scale=self.guidance_embedding_scale,  # use configurable embedding scale
            batch=batch,
        )

        if anchor_embeds is not None:
            class_pred_neg, anchor_pred_neg = pred_negative.chunk(2, dim=0)
        else:
            class_pred_neg = pred_negative
            anchor_pred_neg = None

        # Compute negative losses
        enhance_loss = F.mse_loss(class_pred_neg, enhance_negative_target)
        erase_loss = F.mse_loss(class_pred_neg, erase_positive_target)

        if anchor_target is None:
            anchor_loss = torch.zeros_like(erase_loss)
        else:
            anchor_loss = F.mse_loss(anchor_pred_neg, anchor_target)

        anchor_loss = anchor_loss * self.anchor_strength
        total_neg_loss = (enhance_loss + erase_loss + anchor_loss) / 3.0

        # Send backward for negative multiplier training
        total_neg_loss.backward()
        total_neg_loss = total_neg_loss.detach()

        # Reset multiplier to positive
        network.set_multiplier(1.0)

        # Combine losses for reporting
        total_loss = (total_pos_loss + total_neg_loss) / 2.0

        # Add a grad so backward works right
        total_loss.requires_grad_(True)
        return total_loss

    def is_slider_training_enabled(self) -> bool:
        """Check if slider training is properly configured and enabled."""
        return (
            self.is_initialized
            and bool(self.positive_prompt)
            and bool(self.negative_prompt)
            and bool(self.target_class)
        )

    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration dictionary for saving/loading."""
        return {
            "guidance_strength": self.guidance_strength,
            "anchor_strength": self.anchor_strength,
            "positive_prompt": self.positive_prompt,
            "negative_prompt": self.negative_prompt,
            "target_class": self.target_class,
            "anchor_class": self.anchor_class,
        }
