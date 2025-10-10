"""Main SARA orchestration helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.logger import get_logger

from enhancements.repa.enhanced_repa_helper import EnhancedRepaHelper
from enhancements.repa.encoder_manager import preprocess_raw_image

from .config import SaraConfig
from .autocorrelation import AutocorrelationAligner
from .adversarial import AdversarialAligner
from .loss_aggregator import SaraLossAggregator
from .utils import match_feature_shapes


logger = get_logger(__name__)


class SaraHelper(nn.Module):
    """Combine patch, structural, and adversarial alignment for Takenoko."""

    def __init__(
        self,
        diffusion_model: Any,
        args: Any,
        config: Optional[SaraConfig] = None,
    ) -> None:
        super().__init__()
        config = config or SaraConfig.from_args(args)
        if config is None or not config.enabled:
            raise ValueError("SARA must be enabled before creating SaraHelper")

        self.config = config
        self.args = args
        self.diffusion_model = diffusion_model

        logger.info("Initialising SARA helper with config: %s", config)

        # Ensure underlying REPA helper respects SARA configuration
        setattr(self.args, "model_cache_dir", config.encoder_cache_dir)
        setattr(self.args, "repa_encoder_name", config.encoder_name)
        setattr(self.args, "repa_alignment_depth", config.alignment_depth)

        self.repa_helper = EnhancedRepaHelper(diffusion_model, args)
        self.encoder_feature_dim = max(self.repa_helper.encoder_dims)
        self.diffusion_feature_dim = self.repa_helper.diffusion_hidden_dim

        self.autocorr_aligner = (
            AutocorrelationAligner(config)
            if config.autocorr_loss_weight > 0
            else None
        )
        if self.autocorr_aligner is None:
            logger.info("SARA structural alignment disabled (weight=0)")

        if config.adversarial_enabled and config.adversarial_loss_weight > 0:
            self.adversarial_aligner = AdversarialAligner(
                config, self.encoder_feature_dim
            )
        else:
            self.adversarial_aligner = None
            logger.info("SARA adversarial alignment disabled")

        self.loss_aggregator = SaraLossAggregator(config)
        self.global_step = 0
        self._accelerator_configured = False

    # ---- lifecycle -------------------------------------------------
    def setup_hooks(self) -> None:
        self.repa_helper.setup_hooks()
        logger.info("SARA forward hooks registered")

    def remove_hooks(self) -> None:
        self.repa_helper.remove_hooks()
        logger.info("SARA forward hooks removed")

    def configure_accelerator(self, accelerator: Any) -> None:
        """Allow external accelerators to drive mixed precision."""
        if self._accelerator_configured:
            return
        if self.adversarial_aligner is not None:
            autocast_factory = (
                (lambda: accelerator.autocast())
                if hasattr(accelerator, "autocast")
                else None
            )
            grad_scaler = getattr(accelerator, "scaler", None)
            self.adversarial_aligner.configure_mixed_precision(
                autocast_factory=autocast_factory,
                grad_scaler=grad_scaler,
            )
        self._accelerator_configured = True

    # ---- feature helpers -------------------------------------------
    def _get_encoder_features(
        self,
        clean_pixels: torch.Tensor,
        vae: Optional[Any] = None,
    ) -> List[torch.Tensor]:
        if clean_pixels.dim() == 5:
            clean_pixels = clean_pixels[:, :, 0]

        encoder_features: List[torch.Tensor] = []

        with torch.no_grad():
            images = ((clean_pixels + 1.0) / 2.0).clamp(0, 1) * 255.0
            images = images.to(dtype=torch.float32)
            for encoder, encoder_type in zip(
                self.repa_helper.encoders, self.repa_helper.encoder_types
            ):
                device = next(encoder.parameters()).device
                dtype = next(encoder.parameters()).dtype
                processed = preprocess_raw_image(images, encoder_type).to(
                    device=device, dtype=dtype
                )
                feats = encoder.forward_features(processed)

                if isinstance(feats, dict):
                    if "x_norm_patchtokens" in feats:
                        feats = feats["x_norm_patchtokens"]
                    elif "x_norm_clstoken" in feats:
                        feats = feats["x_norm_clstoken"]
                    else:
                        for value in feats.values():
                            if torch.is_tensor(value):
                                feats = value
                                break

                if feats.dim() == 2:
                    feats = feats.unsqueeze(1)
                elif feats.dim() > 3:
                    bsz, channels, height, width = feats.shape
                    feats = feats.view(bsz, channels, height * width).transpose(1, 2)

                encoder_features.append(feats.to(device=device))

        return encoder_features

    def _get_diffusion_features(self) -> List[Tuple[int, torch.Tensor]]:
        captured: List[Tuple[int, torch.Tensor]] = []
        for idx, feats in enumerate(self.repa_helper.captured_features):
            if feats is not None:
                captured.append((idx, feats))
        return captured

    def _project_diffusion_features(
        self, diffusion_features: torch.Tensor
    ) -> List[torch.Tensor]:
        return self.repa_helper.projection_heads(diffusion_features)

    # ---- loss computation -----------------------------------------
    def compute_sara_loss(
        self,
        clean_pixels: torch.Tensor,
        vae: Optional[Any] = None,
        update_discriminator: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if not torch.is_tensor(clean_pixels):
            raise TypeError("clean_pixels must be a torch.Tensor")
        if clean_pixels.dim() not in {4, 5}:
            raise ValueError(
                f"Expected clean_pixels with 4 or 5 dimensions, got {clean_pixels.dim()}"
            )
        if clean_pixels.dtype not in (
            torch.float16,
            torch.float32,
            torch.bfloat16,
        ):
            raise TypeError(
                f"clean_pixels must be floating point (float16/bfloat16/float32), got {clean_pixels.dtype}"
            )
        if (
            clean_pixels.dim() == 4
            and clean_pixels.shape[1] != 3
        ) or (
            clean_pixels.dim() == 5
            and clean_pixels.shape[1] != 3
        ):
            raise ValueError(
                f"Expected clean_pixels channel dimension to be 3, got {clean_pixels.shape}"
            )
        if clean_pixels.numel() == 0:
            raise ValueError("clean_pixels cannot be empty")
        if not torch.isfinite(clean_pixels).all():
            raise ValueError("clean_pixels contains NaN or infinite values")

        encoder_features_list = self._get_encoder_features(clean_pixels, vae)
        captured_layers = self._get_diffusion_features()

        if not captured_layers:
            logger.warning("SARA: no diffusion features captured this step")
            param_ref = None
            for module in (self.repa_helper, self.diffusion_model, self):
                if not isinstance(module, nn.Module):
                    continue
                param_ref = next(module.parameters(), None)
                if param_ref is not None:
                    break

            device = (
                param_ref.device if param_ref is not None else clean_pixels.device
            )
            dtype = (
                param_ref.dtype if param_ref is not None else clean_pixels.dtype
            )
            return torch.zeros((), device=device, dtype=dtype), {}

        if self.autocorr_aligner is not None:
            self.autocorr_aligner.clear_cache()

        per_layer_patch: List[torch.Tensor] = []
        per_layer_autocorr: List[torch.Tensor] = []
        per_layer_adv: List[torch.Tensor] = []
        metrics: Dict[str, Any] = {}

        for layer_idx, diffusion_features in captured_layers:
            projected_list = self._project_diffusion_features(diffusion_features)

            layer_patch_terms: List[torch.Tensor] = []
            layer_autocorr_terms: List[torch.Tensor] = []
            layer_adv_fake_terms: List[torch.Tensor] = []
            layer_adv_real_terms: List[torch.Tensor] = []

            for projected, encoder_features in zip(
                projected_list, encoder_features_list
            ):
                projected, encoder_features = match_feature_shapes(
                    projected, encoder_features, mode="interpolate"
                )

                if projected.dim() == 2:
                    projected = projected.unsqueeze(1)
                if encoder_features.dim() == 2:
                    encoder_features = encoder_features.unsqueeze(1)

                if self.config.patch_loss_weight > 0:
                    layer_patch_terms.append(
                        self._compute_patch_loss(projected, encoder_features)
                    )

                if self.autocorr_aligner is not None:
                    autocorr = self.autocorr_aligner(
                        pred_features=projected,
                        target_features=encoder_features,
                    )
                    layer_autocorr_terms.append(autocorr["loss"])

                if self.adversarial_aligner is not None:
                    layer_adv_fake_terms.append(projected)
                    layer_adv_real_terms.append(encoder_features)

            if layer_patch_terms:
                per_layer_patch.append(torch.stack(layer_patch_terms).mean())
            if layer_autocorr_terms:
                per_layer_autocorr.append(torch.stack(layer_autocorr_terms).mean())

            if self.adversarial_aligner is not None and layer_adv_fake_terms:
                min_tokens = min(term.shape[1] for term in layer_adv_fake_terms)
                layer_adv_fake_terms = [
                    term[:, :min_tokens, :] for term in layer_adv_fake_terms
                ]
                layer_adv_real_terms = [
                    term[:, :min_tokens, :] for term in layer_adv_real_terms
                ]

                aligner_dim = (
                    self.adversarial_aligner.feature_dim
                    if self.adversarial_aligner is not None
                    else max(term.shape[2] for term in layer_adv_fake_terms)
                )
                target_dim = max(
                    aligner_dim, max(term.shape[2] for term in layer_adv_fake_terms)
                )
                padded_fake = []
                padded_real = []
                for fake_term, real_term in zip(
                    layer_adv_fake_terms, layer_adv_real_terms
                ):
                    # Different encoders can emit slightly different hidden dims. We size
                    # the discriminator to the largest known width and zero-pad the other
                    # streams so we can average them safely. Adaptive pooling would be
                    # heavier here; the zero-fill keeps shapes in sync without learning
                    # fake features because the padded dimensions are never upweighted.
                    if fake_term.shape[2] < target_dim:
                        pad = target_dim - fake_term.shape[2]
                        fake_term = F.pad(fake_term, (0, pad))
                    if real_term.shape[2] < target_dim:
                        pad = target_dim - real_term.shape[2]
                        real_term = F.pad(real_term, (0, pad))
                    padded_fake.append(fake_term)
                    padded_real.append(real_term)

                fake_features = torch.stack(padded_fake).mean(dim=0)
                real_features = torch.stack(padded_real).mean(dim=0)

                step_for_schedule = self.global_step + (
                    1 if update_discriminator else 0
                )
                # We nudge the schedule forward by one step so that the very first
                # discriminator update happens after the warmup completes rather than
                # at global_step==0. This is equivalent to starting the interval on
                # step 1 (e.g. 1, 6, 11 when the interval is 5) and was chosen to
                # mirror the cadence used in the paper's reference implementation.

                if (
                    update_discriminator
                    and self.adversarial_aligner is not None
                    and self.adversarial_aligner.is_past_warmup()
                ):
                    if (
                        step_for_schedule
                        % self.config.discriminator_update_interval
                        == 0
                    ):
                        for _ in range(
                            self.config.discriminator_updates_per_step
                        ):
                            d_metrics = self.adversarial_aligner.update_discriminator(
                                real_features=real_features,
                                fake_features=fake_features,
                            )
                            metrics.update(
                                {f"sara_{k}": v for k, v in d_metrics.items()}
                            )

                adv_loss, g_metrics = self.adversarial_aligner.compute_generator_loss(
                    fake_features=fake_features,
                    real_features=real_features,
                )
                per_layer_adv.append(adv_loss)
                metrics.update({f"sara_{k}": v for k, v in g_metrics.items()})

        patch_loss = (
            torch.stack(per_layer_patch).mean() if per_layer_patch else None
        )
        autocorr_loss = (
            torch.stack(per_layer_autocorr).mean()
            if per_layer_autocorr
            else None
        )
        adversarial_loss = (
            torch.stack(per_layer_adv).mean() if per_layer_adv else None
        )

        if patch_loss is not None:
            metrics["sara_patch_loss"] = float(patch_loss.detach().item())
        if autocorr_loss is not None:
            metrics["sara_autocorr_loss"] = float(autocorr_loss.detach().item())
        if adversarial_loss is not None:
            metrics["sara_adversarial_loss"] = float(
                adversarial_loss.detach().item()
            )

        components = self.loss_aggregator.aggregate(
            patch_loss=patch_loss,
            autocorr_loss=autocorr_loss,
            adversarial_loss=adversarial_loss,
        )
        metrics.update({f"sara_{k}": v for k, v in components.to_dict().items()})

        self.repa_helper.captured_features = [
            None
            for _ in self.repa_helper.captured_features
        ]

        if update_discriminator:
            self.global_step += 1
            if self.adversarial_aligner is not None:
                self.adversarial_aligner.increment_step()

        return components.total_loss, metrics

    def _compute_patch_loss(
        self,
        projected_features: torch.Tensor,
        encoder_features: torch.Tensor,
    ) -> torch.Tensor:
        sim_fn = self.config.similarity_fn

        if sim_fn == "cosine":
            proj = F.normalize(projected_features, dim=-1)
            target = F.normalize(encoder_features, dim=-1)
            similarity = (proj * target).sum(dim=-1)
            return -similarity.mean()

        if sim_fn == "mse":
            return F.mse_loss(projected_features, encoder_features)

        if sim_fn == "l1":
            return F.l1_loss(projected_features, encoder_features)

        raise ValueError(f"Unsupported similarity function: {sim_fn}")

    def get_sara_loss(
        self,
        clean_pixels: torch.Tensor,
        vae: Optional[Any] = None,
    ) -> torch.Tensor:
        loss, _ = self.compute_sara_loss(
            clean_pixels,
            vae=vae,
            update_discriminator=True,
        )
        return loss

    @torch.no_grad()
    def get_sara_metrics(
        self,
        clean_pixels: torch.Tensor,
        vae: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Expose detailed metrics for parity with REPA helper."""
        loss, metrics = self.compute_sara_loss(
            clean_pixels,
            vae=vae,
            update_discriminator=False,
        )
        metrics["sara_total_loss"] = float(loss.detach().item())
        return metrics

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        state = super().state_dict(*args, **kwargs)
        state["config"] = self.config.__dict__
        state["repa_helper"] = self.repa_helper.state_dict()
        state["global_step"] = self.global_step
        if self.autocorr_aligner is not None:
            state["autocorr_aligner"] = self.autocorr_aligner.state_dict()
        if self.adversarial_aligner is not None:
            state["adversarial_aligner"] = self.adversarial_aligner.state_dict()
        return state

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]  # noqa: ANN001
        self.repa_helper.load_state_dict(
            state_dict["repa_helper"], strict=strict
        )
        self.global_step = state_dict.get("global_step", 0)

        if self.autocorr_aligner is not None and "autocorr_aligner" in state_dict:
            self.autocorr_aligner.load_state_dict(
                state_dict["autocorr_aligner"], strict=strict
            )
        if (
            self.adversarial_aligner is not None
            and "adversarial_aligner" in state_dict
        ):
            self.adversarial_aligner.load_state_dict(
                state_dict["adversarial_aligner"], strict=strict
            )
        logger.info("SARA helper state restored")


def create_sara_helper(
    diffusion_model: Any,
    args: Any,
) -> Optional[SaraHelper]:
    config = SaraConfig.from_args(args)
    if config is None or not config.enabled:
        logger.info("SARA disabled. Skipping helper creation.")
        return None

    helper = SaraHelper(diffusion_model, args, config=config)
    logger.info("SARA helper created")
    return helper
