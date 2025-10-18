"""Training loss computation utilities.

This module encapsulates all logic related to computing the training loss, so the
main training loop can remain focused on orchestration. It preserves behavior from
the previous inlined implementation in `core/training_core.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Callable, List

import torch
import torch.nn.functional as F

from common.logger import get_logger
from criteria.dispersive_loss import dispersive_loss_info_nce
from criteria.loss_factory import conditional_loss_with_pseudo_huber
from core.masked_training_manager import (
    create_masked_training_manager,
    MaskedTrainingManager,
)
from utils.train_utils import get_sigmas


logger = get_logger(__name__)


def _get_loss_kwargs(args):
    """Extract loss-specific parameters from args to pass to loss functions."""
    return {
        # Fourier loss parameters
        "fourier_mode": getattr(args, "fourier_mode", "weighted"),
        "fourier_dims": getattr(args, "fourier_dims", (-2, -1)),
        "fourier_eps": getattr(args, "fourier_eps", 1e-8),
        # Note: fourier_normalize removed as not supported by fourier functions
        "fourier_multiscale_factors": getattr(
            args, "fourier_multiscale_factors", [1, 2, 4]
        ),
        "fourier_adaptive_threshold": getattr(args, "fourier_adaptive_threshold", 0.1),
        "fourier_adaptive_alpha": getattr(args, "fourier_adaptive_alpha", 0.5),
        "fourier_high_freq_weight": getattr(args, "fourier_high_freq_weight", 2.0),
        # Wavelet parameters
        "wavelet_type": getattr(args, "wavelet_type", "haar"),
        "wavelet_levels": getattr(args, "wavelet_levels", 1),
        "wavelet_mode": getattr(args, "wavelet_mode", "zero"),
        # Clustered MSE parameters
        "clustered_mse_num_clusters": getattr(args, "clustered_mse_num_clusters", 8),
        "clustered_mse_cluster_weight": getattr(
            args, "clustered_mse_cluster_weight", 1.0
        ),
        # Huber parameters
        "huber_delta": getattr(args, "huber_delta", 1.0),
        # EW loss parameters
        "ew_boundary_shift": getattr(args, "ew_boundary_shift", 0.0),
        # Stepped loss parameters
        "stepped_step_size": getattr(args, "stepped_step_size", 50),
        "stepped_multiplier": getattr(args, "stepped_multiplier", 10.0),
    }


@dataclass
class LossComponents:
    """Container for individual loss terms and the final total loss.

    Attributes
    ----------
    total_loss: torch.Tensor
        The final scalar loss to backpropagate.
    base_loss: Optional[torch.Tensor]
        The base flow-matching loss (after reductions and any weighting).
    dop_loss: Optional[torch.Tensor]
        The Diff Output Preservation loss component, if enabled.
    dispersive_loss: Optional[torch.Tensor]
        The dispersive (InfoNCE-style) loss component, if enabled.
    optical_flow_loss: Optional[torch.Tensor]
        The optical flow consistency loss, if enabled.
    repa_loss: Optional[torch.Tensor]
        The REPA alignment loss, if enabled.
    """

    total_loss: torch.Tensor
    base_loss: Optional[torch.Tensor] = None
    dop_loss: Optional[torch.Tensor] = None
    dispersive_loss: Optional[torch.Tensor] = None
    optical_flow_loss: Optional[torch.Tensor] = None
    repa_loss: Optional[torch.Tensor] = None
    sara_loss: Optional[torch.Tensor] = None
    ortho_reg_p: Optional[torch.Tensor] = None
    ortho_reg_q: Optional[torch.Tensor] = None


class TrainingLossComputer:
    """Compute training losses with feature flags preserved via `args`.

    Parameters
    ----------
    config: Any
        Configuration object providing model/patch settings. Must expose
        `patch_size` as a tuple (pt, ph, pw).
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self._last_contrastive_components = None
        # Add masked training manager (initialized later with args)
        self._masked_training_manager: Optional[MaskedTrainingManager] = None

    # ---- Internal helpers ----
    def _compute_seq_len(self, latents: torch.Tensor) -> int:
        lat_f, lat_h, lat_w = latents.shape[2:5]
        pt, ph, pw = self.config.patch_size
        return (lat_f * lat_h * lat_w) // (pt * ph * pw)

    def _maybe_get_control_latents(
        self,
        args: Any,
        accelerator: Any,
        batch: Dict[str, Any],
        latents: torch.Tensor,
        network_dtype: torch.dtype,
        vae: Optional[Any],
        control_signal_processor: Optional[Any],
    ) -> Optional[torch.Tensor]:
        if (
            hasattr(args, "enable_control_lora")
            and args.enable_control_lora
            and control_signal_processor is not None
            and hasattr(control_signal_processor, "process_control_signal")
        ):
            try:
                return control_signal_processor.process_control_signal(
                    args, accelerator, batch, latents, network_dtype, vae
                )
            except Exception as e:
                logger.warning(f"Control signal processing failed for DOP path: {e}")
                return None
        return None

    def _concat_control_if_available(
        self,
        noisy_model_input: torch.Tensor,
        control_latents: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if control_latents is None:
            return noisy_model_input
        concat_dim = 1 if noisy_model_input.dim() == 5 else 0
        return torch.cat(
            [noisy_model_input, control_latents.to(noisy_model_input)], dim=concat_dim
        )

    @torch.no_grad()
    def _compute_prior_pred_for_dop(
        self,
        args: Any,
        accelerator: Any,
        transformer: Any,
        network: Any,
        latents: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        dop_context: List[torch.Tensor],
        model_input_control: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the base model prediction with LoRA disabled for DOP.

        Returns a detached tensor suitable for MSE target.
        """
        # Temporarily disable LoRA
        original_multiplier = accelerator.unwrap_model(network).multiplier
        accelerator.unwrap_model(network).multiplier = 0.0
        try:
            seq_len = self._compute_seq_len(latents)
            model_input_prior = (
                model_input_control
                if model_input_control is not None
                else noisy_model_input
            )
            prior_pred_list = transformer(
                model_input_prior,
                t=timesteps,
                context=dop_context,
                seq_len=seq_len,
                y=None,
            )
            prior_pred = torch.stack(prior_pred_list, dim=0).detach()
            return prior_pred
        finally:
            # Restore
            accelerator.unwrap_model(network).multiplier = original_multiplier

    def initialize_masked_training(self, args) -> None:
        """Initialize masked training manager from args."""
        self._masked_training_manager = create_masked_training_manager(args)
        if self._masked_training_manager:
            logger.info("Masked training with prior preservation enabled")

    def _compute_prior_prediction_if_needed(
        self,
        args,
        accelerator,
        transformer,
        network,
        latents,
        noisy_model_input,
        timesteps,
        batch,
        network_dtype,
    ) -> Optional[torch.Tensor]:
        """Compute prior prediction for masked training if enabled."""
        if (
            self._masked_training_manager is None
            or self._masked_training_manager.config.masked_prior_preservation_weight
            <= 0
        ):
            return None

        if transformer is None or network is None:
            logger.warning(
                "Cannot compute prior prediction: transformer or network is None"
            )
            return None

        try:
            # Temporarily disable LoRA
            original_multiplier = accelerator.unwrap_model(network).multiplier
            accelerator.unwrap_model(network).multiplier = 0.0

            with torch.no_grad():
                # Compute sequence length for transformer
                seq_len = self._compute_seq_len(latents)

                # Use Takenoko's actual context structure - check multiple possible keys
                context = batch.get("text_encoder_hidden_states")
                if context is None:
                    context = batch.get("encoder_hidden_states")
                if context is None:
                    context = batch.get("context")

                # Ensure context is in correct format for Takenoko
                if context is not None:
                    if not isinstance(context, list):
                        context = [context] if context is not None else []
                else:
                    context = []

                # Get prior prediction using Takenoko's transformer interface
                prior_pred_list = transformer(
                    noisy_model_input,
                    t=timesteps,
                    context=context,
                    seq_len=seq_len,
                    y=batch.get("y", None),  # Include y parameter if available
                )
                prior_pred = torch.stack(prior_pred_list, dim=0).detach()
                return prior_pred

        except Exception as e:
            logger.warning(f"Failed to compute prior prediction: {e}")
            return None
        finally:
            # Restore LoRA multiplier
            if "original_multiplier" in locals():
                accelerator.unwrap_model(network).multiplier = original_multiplier

    def compute_training_loss(
        self,
        *,
        args: Any,
        accelerator: Any,
        latents: torch.Tensor,
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
        model_pred: torch.Tensor,
        target: torch.Tensor,
        weighting: Optional[torch.Tensor],
        batch: Dict[str, Any],
        intermediate_z: Optional[torch.Tensor],
        vae: Optional[Any] = None,
        transformer: Optional[Any] = None,
        network: Optional[Any] = None,
        control_signal_processor: Optional[Any] = None,
        repa_helper: Optional[Any] = None,
        sara_helper: Optional[Any] = None,
        raft: Optional[Any] = None,
        warp_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        adaptive_manager: Optional[Any] = None,
        transition_loss_context: Optional[Dict[str, Any]] = None,
        noise_scheduler: Optional[Any] = None,
    ) -> LossComponents:
        """Compute the full training loss and its components.

        The returned `total_loss` is the scalar to backpropagate.
        """

        if transition_loss_context is None:
            transition_loss_context = {"transition_enabled": False}
        transition_enabled = bool(
            transition_loss_context.get("transition_training_enabled")
        )
        transition_directional_tensor = None
        transition_directional_weight = 0.0
        if transition_enabled:
            transition_directional_tensor = transition_loss_context.get("directional_loss")
            try:
                transition_directional_weight = float(
                    transition_loss_context.get("directional_weight", 0.0)
                )
            except Exception:
                transition_directional_weight = 0.0

        # ---- Contrastive Flow Matching (Enhanced with DeltaFM improvements) ----
        if (
            hasattr(args, "enable_contrastive_flow_matching")
            and args.enable_contrastive_flow_matching
        ):
            # Import enhanced utilities only when needed for backward compatibility
            try:
                from common.contrastive_flow_utils import (
                    class_conditioned_negative_sampling,
                    compute_enhanced_contrastive_loss,
                    extract_class_labels_from_batch,
                )

                use_enhanced = True
            except ImportError:
                logger.warning(
                    "Enhanced contrastive flow utilities not available, using basic implementation"
                )
                use_enhanced = False

            batch_size = latents.size(0)
            current_step = getattr(args, "current_step", None)
            total_steps = getattr(args, "total_steps", None)
            lambda_val = float(getattr(args, "contrastive_flow_lambda", 0.05))

            if use_enhanced:
                # Enhanced implementation with class-conditioned sampling
                labels = extract_class_labels_from_batch(batch)
                use_class_conditioning = (
                    getattr(args, "contrastive_flow_class_conditioning", True)
                    and labels is not None
                )

                if use_class_conditioning:
                    try:
                        negative_indices = class_conditioned_negative_sampling(
                            labels, accelerator.device
                        )
                    except (ValueError, RuntimeError) as e:
                        logger.warning(
                            f"Class-conditioned sampling failed ({e}), falling back to random sampling"
                        )
                        negative_indices = torch.randperm(
                            batch_size, device=accelerator.device
                        )
                        use_class_conditioning = False  # Update for monitoring
                else:
                    negative_indices = torch.randperm(
                        batch_size, device=accelerator.device
                    )

                negative_latents = latents[negative_indices]
                negative_noise = noise[negative_indices]
                negative_target = negative_noise - negative_latents.to(
                    device=accelerator.device, dtype=network_dtype
                )

                def loss_fn(pred, tgt, **kwargs):
                    loss_kwargs = _get_loss_kwargs(args)
                    loss_kwargs.update(kwargs)

                    return conditional_loss_with_pseudo_huber(
                        pred,
                        tgt,
                        loss_type=args.loss_type,
                        huber_c=args.pseudo_huber_c,
                        current_step=current_step,
                        total_steps=total_steps,
                        schedule_type=args.pseudo_huber_schedule_type,
                        c_min=args.pseudo_huber_c_min,
                        c_max=args.pseudo_huber_c_max,
                        reduction="none",
                        timesteps=timesteps,
                        noise=noise,
                        noisy_latents=noisy_model_input,
                        clean_latents=latents,
                        noise_scheduler=noise_scheduler,
                        **loss_kwargs,
                    )

                contrastive_result = compute_enhanced_contrastive_loss(
                    model_pred=model_pred.to(network_dtype),
                    target=target,
                    negative_target=negative_target,
                    labels=labels,
                    null_class_idx=getattr(
                        args, "contrastive_flow_null_class_idx", None
                    ),
                    dont_contrast_on_unconditional=getattr(
                        args, "contrastive_flow_skip_unconditional", False
                    ),
                    lambda_val=lambda_val,
                    loss_fn=loss_fn,
                )

                loss = contrastive_result["total_loss"]
                self._last_contrastive_components = {
                    "flow_loss": contrastive_result["flow_loss"].mean().item(),
                    "contrastive_loss": contrastive_result["contrastive_loss"]
                    .mean()
                    .item(),
                    "lambda_val": lambda_val,
                    "used_class_conditioning": use_class_conditioning,
                }
            else:
                # Basic implementation (original Takenoko behavior)
                shuffled_indices = torch.randperm(batch_size, device=accelerator.device)
                negative_latents = latents[shuffled_indices]
                negative_noise = noise[shuffled_indices]
                negative_target = negative_noise - negative_latents.to(
                    device=accelerator.device, dtype=network_dtype
                )

                loss_kwargs = _get_loss_kwargs(args)

                loss_fm = conditional_loss_with_pseudo_huber(
                    model_pred.to(network_dtype),
                    target,
                    loss_type=args.loss_type,
                    huber_c=args.pseudo_huber_c,
                    current_step=current_step,
                    total_steps=total_steps,
                    schedule_type=args.pseudo_huber_schedule_type,
                    c_min=args.pseudo_huber_c_min,
                    c_max=args.pseudo_huber_c_max,
                    reduction="none",
                    timesteps=timesteps,
                    noise=noise,
                    noisy_latents=noisy_model_input,
                    clean_latents=latents,
                    noise_scheduler=noise_scheduler,
                    **loss_kwargs,
                )
                loss_contrastive = conditional_loss_with_pseudo_huber(
                    model_pred.to(network_dtype),
                    negative_target,
                    loss_type=args.loss_type,
                    huber_c=args.pseudo_huber_c,
                    current_step=current_step,
                    total_steps=total_steps,
                    schedule_type=args.pseudo_huber_schedule_type,
                    c_min=args.pseudo_huber_c_min,
                    c_max=args.pseudo_huber_c_max,
                    reduction="none",
                    timesteps=timesteps,
                    noise=noise,
                    noisy_latents=noisy_model_input,
                    clean_latents=latents,
                    noise_scheduler=noise_scheduler,
                    **loss_kwargs,
                )

                loss = loss_fm - lambda_val * loss_contrastive
                self._last_contrastive_components = {
                    "flow_loss": loss_fm.mean().item(),
                    "contrastive_loss": loss_contrastive.mean().item(),
                    "lambda_val": lambda_val,
                    "used_class_conditioning": False,
                }
        else:
            # Use conditional loss function based on loss_type
            current_step = getattr(args, "current_step", None)
            total_steps = getattr(args, "total_steps", None)

            loss = conditional_loss_with_pseudo_huber(
                model_pred.to(network_dtype),
                target,
                loss_type=args.loss_type,
                huber_c=args.pseudo_huber_c,
                current_step=current_step,
                total_steps=total_steps,
                schedule_type=args.pseudo_huber_schedule_type,
                c_min=args.pseudo_huber_c_min,
                c_max=args.pseudo_huber_c_max,
                reduction="none",
                timesteps=timesteps,
                noise=noise,
                noisy_latents=noisy_model_input,
                clean_latents=latents,
                noise_scheduler=noise_scheduler,
                **_get_loss_kwargs(args),
            )

        # ---- Dataset sample weights ----
        sample_weights = batch.get("weight", None)
        if sample_weights is not None:
            sample_weights = sample_weights.to(
                device=accelerator.device, dtype=network_dtype
            )
            while sample_weights.dim() < loss.dim():
                sample_weights = sample_weights.unsqueeze(-1)
            loss = loss * sample_weights

        # ---- Masked training ----
        mask = batch.get("mask_signal", None)
        if mask is not None:
            if self._masked_training_manager is not None:
                # Use enhanced masked training with prior preservation
                # Check if DOP has already computed a prior prediction to avoid duplication
                prior_pred = None
                if (
                    getattr(args, "diff_output_preservation", False)
                    and "t5_preservation" in batch
                    and transformer is not None
                    and network is not None
                ):
                    # DOP is active - we'll compute DOP's prior and reuse it for masking
                    logger.debug(
                        "DOP active: deferring prior computation to DOP section"
                    )
                else:
                    # No DOP conflict - compute prior for masking only
                    prior_pred = self._compute_prior_prediction_if_needed(
                        args,
                        accelerator,
                        transformer,
                        network,
                        latents,
                        noisy_model_input,
                        timesteps,
                        batch,
                        network_dtype,
                    )

                # Replace basic masking with enhanced version
                loss = self._masked_training_manager.compute_masked_loss_with_prior(
                    model_pred=model_pred.to(network_dtype),
                    target=target,
                    mask=mask,
                    prior_pred=prior_pred,  # Will be None if DOP is active
                    loss_type=getattr(args, "loss_type", "mse"),
                    huber_c=getattr(args, "pseudo_huber_c", 1.0),
                    current_step=getattr(args, "current_step", None),
                    total_steps=getattr(args, "total_steps", None),
                    schedule_type=getattr(
                        args, "pseudo_huber_schedule_type", "constant"
                    ),
                    c_min=getattr(args, "pseudo_huber_c_min", 0.01),
                    c_max=getattr(args, "pseudo_huber_c_max", 10.0),
                )
                base_loss = loss  # Update base_loss for consistency
            else:
                # Existing basic masking (unchanged for backward compatibility)
                mask = mask.to(device=accelerator.device, dtype=network_dtype)
                while mask.dim() < loss.dim():
                    mask = mask.unsqueeze(-1)
                loss = loss * mask

        if weighting is not None:
            loss = loss * weighting

        # ---- Apply adaptive timestep importance weights ----
        if adaptive_manager is not None and adaptive_manager.enabled:
            try:
                # Convert timesteps to 0-1 range for importance lookup
                timesteps_normalized = timesteps.float() / 1000.0

                # Get importance weights
                importance_weights = adaptive_manager.get_adaptive_sampling_weights(
                    timesteps_normalized
                )

                if importance_weights is not None and importance_weights.numel() > 0:
                    # Ensure weights have correct shape for broadcasting
                    if importance_weights.dim() > 1:
                        importance_weights = importance_weights.view(-1)

                    # Reshape loss to match batch dimension if needed
                    if loss.dim() > 1:
                        # Reduce loss to per-sample while preserving batch dimension
                        loss_per_sample = loss.view(loss.size(0), -1).mean(dim=1)
                    else:
                        loss_per_sample = loss

                    # Apply importance weights
                    if loss_per_sample.size(0) == importance_weights.size(0):
                        loss = loss_per_sample * importance_weights
                        logger.debug(
                            f"Applied adaptive importance weights: mean={importance_weights.mean():.3f}"
                        )
                    else:
                        logger.debug(
                            f"Size mismatch: loss {loss_per_sample.size(0)} vs weights {importance_weights.size(0)}"
                        )

            except Exception as e:
                logger.debug(f"Error applying adaptive importance weights: {e}")
                # Continue without adaptive weighting

        # ---- Explicit video-aware loss reduction
        use_explicit = getattr(args, "use_explicit_video_loss_reduction", False)
        if use_explicit and loss.dim() > 1:
            # Dimension-aware reduction: handle video (5D) vs image (4D) tensors explicitly
            if len(model_pred.shape) == 5:
                # Video: (B, C, F, H, W) -> reduce [1, 2, 3, 4]
                loss = loss.mean([1, 2, 3, 4])
            else:
                # Image: (B, C, H, W) -> reduce [1, 2, 3]
                loss = loss.mean([1, 2, 3])
            # Final batch reduction
            loss = loss.mean()
        else:
            # Default behavior: reduce all dimensions at once
            loss = loss.mean()
        if (
            transition_enabled
            and transition_directional_tensor is not None
            and transition_directional_weight > 0.0
            and isinstance(transition_directional_tensor, torch.Tensor)
        ):
            directional_value = transition_directional_tensor.to(
                loss.device, loss.dtype
            )
            if directional_value.dim() > 1:
                directional_value = directional_value.view(
                    directional_value.size(0), -1
                ).mean(dim=1)
            loss = loss + transition_directional_weight * directional_value.mean()

        base_loss = loss

        # ---- Optional Dispersive Loss ----
        dispersive_loss_value: Optional[torch.Tensor] = None
        if (
            hasattr(args, "enable_dispersive_loss")
            and args.enable_dispersive_loss
            and intermediate_z is not None
            and float(getattr(args, "dispersive_loss_lambda", 0.0)) != 0.0
        ):
            try:
                pooled_z = intermediate_z
                pooling_mode = str(
                    getattr(args, "dispersive_loss_pooling", "none")
                ).lower()
                if pooling_mode != "none":
                    try:
                        lat_f, lat_h, lat_w = latents.shape[2:5]
                        pt, ph, pw = self.config.patch_size
                        t_tokens = max(1, lat_f // pt)
                        h_tokens = max(1, lat_h // ph)
                        w_tokens = max(1, lat_w // pw)
                        tokens_per_frame = h_tokens * w_tokens
                        bsz, seq_len, hidden = pooled_z.shape
                        if seq_len == t_tokens * tokens_per_frame:
                            pooled_z = pooled_z.view(
                                bsz, t_tokens, tokens_per_frame, hidden
                            )
                            if pooling_mode == "frame_mean":
                                pooled_z = pooled_z.mean(dim=2)  # (B, T, C)
                                pooled_z = pooled_z.reshape(bsz, -1)
                    except Exception:
                        pooled_z = intermediate_z

                dispersive_val = dispersive_loss_info_nce(
                    pooled_z,
                    tau=float(getattr(args, "dispersive_loss_tau", 0.5)),
                    metric=str(getattr(args, "dispersive_loss_metric", "l2_sq")),
                )
                loss = (
                    loss
                    + float(getattr(args, "dispersive_loss_lambda", 0.0))
                    * dispersive_val
                )
                dispersive_loss_value = dispersive_val.detach()
            except Exception as e:
                logger.warning(f"Dispersive loss computation failed: {e}")

        # ---- Optional SARA or REPA Loss ----
        sara_loss_value: Optional[torch.Tensor] = None
        repa_loss_value: Optional[torch.Tensor] = None
        if sara_helper is not None:
            try:
                if "pixels" in batch:
                    clean_pixels = torch.stack(batch["pixels"], dim=0)
                    first_frame_pixels = clean_pixels[:, :, 0, :, :]
                    sara_val, _ = sara_helper.compute_sara_loss(
                        first_frame_pixels,
                        vae,
                        update_discriminator=accelerator.sync_gradients,
                    )
                    loss = loss + sara_val
                    sara_loss_value = sara_val.detach()
                else:
                    logger.warning(
                        "SARA enabled, but no 'pixels' found in batch. Skipping SARA loss."
                    )
            except Exception as e:
                logger.warning(f"SARA loss computation failed: {e}")
        elif repa_helper is not None:
            try:
                if "pixels" in batch:
                    clean_pixels = torch.stack(
                        batch["pixels"], dim=0
                    )  # (B, C, F, H, W)
                    first_frame_pixels = clean_pixels[:, :, 0, :, :]
                    repa_val = repa_helper.get_repa_loss(first_frame_pixels, vae)
                    loss = loss + repa_val
                    repa_loss_value = repa_val.detach()
                else:
                    logger.warning(
                        "REPA enabled, but no 'pixels' found in batch. Skipping REPA loss."
                    )
            except Exception as e:
                logger.warning(f"REPA loss computation failed: {e}")

        # ---- Optional Optical Flow Loss (RAFT) ----
        optical_flow_loss_value: Optional[torch.Tensor] = None
        if (
            hasattr(args, "enable_optical_flow_loss")
            and args.enable_optical_flow_loss
            and float(getattr(args, "lambda_optical_flow", 0.0)) > 0.0
        ):
            try:
                assert vae is not None, "VAE must be provided for optical flow loss"
                with torch.no_grad():
                    pred_latents = model_pred.to(network_dtype)
                    decoded = vae.decode(
                        pred_latents / getattr(vae, "scaling_factor", 1.0)
                    )
                    pred_frames = (
                        torch.stack(decoded, dim=0)
                        if isinstance(decoded, list)
                        else decoded
                    )
                assert (
                    pred_frames.dim() == 5
                ), f"Expected pred_frames shape (B, T, C, H, W), got {pred_frames.shape}"
                bsz, t_frames, c, h, w = pred_frames.shape
                if t_frames < 2:
                    raise ValueError("Need at least 2 frames for optical flow loss")
                assert (
                    raft is not None
                ), "RAFT model must be available for optical flow loss"
                assert (
                    warp_fn is not None
                ), "Warp function must be provided for optical flow loss"
                with torch.no_grad():
                    frame0 = pred_frames[:, :-1].reshape(-1, c, h, w)
                    frame1 = pred_frames[:, 1:].reshape(-1, c, h, w)
                    flow = raft(frame0, frame1)
                warped = warp_fn(frame0, flow)
                flow_loss = F.l1_loss(warped, frame1)
                loss = loss + float(args.lambda_optical_flow) * flow_loss
                optical_flow_loss_value = flow_loss.detach()
            except Exception as e:
                logger.warning(f"Optical flow loss computation failed: {e}")

        # ---- Diff Output Preservation (DOP) ----
        dop_loss_value: Optional[torch.Tensor] = None
        if (
            getattr(args, "diff_output_preservation", False)
            and "t5_preservation" in batch
            and transformer is not None
            and network is not None
        ):
            try:
                dop_embeds = [
                    t.to(device=accelerator.device, dtype=network_dtype)
                    for t in batch["t5_preservation"]
                ]
                # Control-aware inputs, mirroring training path
                control_latents_dop = self._maybe_get_control_latents(
                    args,
                    accelerator,
                    batch,
                    latents,
                    network_dtype,
                    vae,
                    control_signal_processor,
                )
                model_input_control = None
                if control_latents_dop is not None:
                    model_input_control = self._concat_control_if_available(
                        noisy_model_input, control_latents_dop
                    )

                # Base model (LoRA disabled) prior prediction
                prior_pred = self._compute_prior_pred_for_dop(
                    args=args,
                    accelerator=accelerator,
                    transformer=transformer,
                    network=network,
                    latents=latents,
                    noisy_model_input=noisy_model_input,
                    timesteps=timesteps,
                    dop_context=dop_embeds,
                    model_input_control=model_input_control,
                )

                # Apply masked training to DOP prior if both are enabled
                # This ensures DOP's prior is also subject to masking constraints
                mask = batch.get("mask_signal", None)
                if (
                    mask is not None
                    and self._masked_training_manager is not None
                    and self._masked_training_manager.config.masked_prior_preservation_weight
                    > 0
                ):

                    logger.debug("Applying masked training to DOP prior prediction")
                    # Apply masking to the DOP prior prediction using base model as "prior"
                    # This creates a DOP-aware masked prior that respects both constraints
                    masked_dop_prior = (
                        self._masked_training_manager.compute_masked_loss_with_prior(
                            model_pred=prior_pred,
                            target=target,
                            mask=mask,
                            prior_pred=None,  # No recursive prior for DOP's own prior
                            loss_type=getattr(args, "loss_type", "mse"),
                            huber_c=getattr(args, "pseudo_huber_c", 1.0),
                            current_step=getattr(args, "current_step", None),
                            total_steps=getattr(args, "total_steps", None),
                            schedule_type=getattr(
                                args, "pseudo_huber_schedule_type", "constant"
                            ),
                            c_min=getattr(args, "pseudo_huber_c_min", 0.01),
                            c_max=getattr(args, "pseudo_huber_c_max", 10.0),
                        )
                    )
                    # Note: We don't replace base_loss here since DOP has its own loss component

                # LoRA-enabled prediction on preservation prompt
                seq_len = self._compute_seq_len(latents)
                model_input_dop = (
                    model_input_control
                    if model_input_control is not None
                    else noisy_model_input
                )
                # Match original behavior: compute DOP prediction under autocast
                with accelerator.autocast():
                    dop_pred_list = transformer(
                        model_input_dop,
                        t=timesteps,
                        context=dop_embeds,
                        seq_len=seq_len,
                        y=None,
                    )
                dop_pred = torch.stack(dop_pred_list, dim=0)
                dop_loss_val = F.mse_loss(dop_pred, prior_pred) * float(
                    getattr(args, "diff_output_preservation_multiplier", 1.0)
                )
                loss = loss + dop_loss_val
                dop_loss_value = dop_loss_val.detach()
            except Exception as e:
                logger.warning(f"DOP loss computation failed: {e}")

        # ---- Optional Orthogonal LoRA regularization (T-LoRA orthogonal mode) ----
        ortho_p_val: Optional[torch.Tensor] = None
        ortho_q_val: Optional[torch.Tensor] = None
        try:
            if network is not None:
                lam_p = float(getattr(network, "ortho_reg_lambda_p", 0.0))
                lam_q = float(getattr(network, "ortho_reg_lambda_q", 0.0))
                if (lam_p > 0.0 or lam_q > 0.0) and hasattr(network, "unet_loras"):
                    p_sum: Optional[torch.Tensor] = None
                    q_sum: Optional[torch.Tensor] = None
                    for lora in getattr(network, "unet_loras", []):
                        if hasattr(lora, "_ortho_enabled") and getattr(
                            lora, "_ortho_enabled", False
                        ):
                            if hasattr(lora, "regularization"):
                                try:
                                    p_reg, q_reg = lora.regularization()  # type: ignore
                                except Exception:
                                    p_reg, q_reg = None, None
                            else:
                                p_reg, q_reg = None, None
                            if p_reg is not None:
                                p_sum = p_reg if p_sum is None else p_sum + p_reg
                            if q_reg is not None:
                                q_sum = q_reg if q_sum is None else q_sum + q_reg
                    if p_sum is not None and lam_p > 0.0:
                        loss = loss + lam_p * p_sum
                        ortho_p_val = p_sum.detach()
                    if q_sum is not None and lam_q > 0.0:
                        loss = loss + lam_q * q_sum
                        ortho_q_val = q_sum.detach()
        except Exception as e:
            logger.warning(f"Orthogonal LoRA regularization failed: {e}")

        return LossComponents(
            total_loss=loss,
            base_loss=base_loss.detach(),
            dop_loss=dop_loss_value,
            dispersive_loss=dispersive_loss_value,
            optical_flow_loss=optical_flow_loss_value,
            repa_loss=repa_loss_value,
            sara_loss=sara_loss_value,
            ortho_reg_p=ortho_p_val,
            ortho_reg_q=ortho_q_val,
        )

    @torch.no_grad()
    def compute_extra_train_metrics(
        self,
        *,
        model_pred: torch.Tensor,
        target: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        noise_scheduler: Any,
        accelerator: Any,
    ) -> Dict[str, float]:
        """Compute periodic extra training loss metrics using existing tensors.

        Returns
        -------
        Dict[str, float]
            A mapping with keys:
            - "train/loss_p50"
            - "train/loss_p90"
            - "train/loss_cv_in_batch"
            - "train/direct_noise_loss_mean"
            - "train/loss_snr_correlation_batch"
        """
        metrics: Dict[str, float] = {}
        try:
            per_sample_vel = torch.nn.functional.mse_loss(
                model_pred.to(torch.float32), target.to(torch.float32), reduction="none"
            ).mean(dim=[1, 2, 3, 4])

            # Percentiles
            try:
                p50 = torch.quantile(per_sample_vel, 0.5).item()
            except Exception:
                p50 = torch.median(per_sample_vel).item()
            try:
                p90 = torch.quantile(per_sample_vel, 0.9).item()
            except Exception:
                k = max(1, int(0.1 * per_sample_vel.numel()))
                p90 = per_sample_vel.topk(k).values.min().item()

            # CV (robust to batch size 1; use population std to avoid ddof warnings)
            mean_loss = per_sample_vel.mean()
            batch_items = int(per_sample_vel.numel())
            if batch_items > 1:
                std_loss = per_sample_vel.to(torch.float32).std(correction=0)
                cv_in_batch = (
                    (std_loss / (mean_loss + 1e-12)).item()
                    if torch.isfinite(mean_loss)
                    else 0.0
                )
            else:
                std_loss = torch.tensor(
                    0.0, device=per_sample_vel.device, dtype=torch.float32
                )
                cv_in_batch = 0.0

            # Direct noise loss mean
            direct_noise_loss_mean = (
                torch.nn.functional.mse_loss(
                    model_pred.to(torch.float32),
                    noise.to(torch.float32),
                    reduction="none",
                )
                .mean(dim=[1, 2, 3, 4])
                .mean()
                .item()
            )

            # SNR correlation with loss and slope proxy
            try:
                sigmas = get_sigmas(
                    noise_scheduler,
                    timesteps,
                    accelerator.device,
                    n_dim=4,
                    dtype=timesteps.dtype,
                    source="training/metrics",
                )
                if sigmas.dim() > 1:
                    sigmas_reduced = sigmas.view(sigmas.shape[0], -1).mean(dim=1)
                else:
                    sigmas_reduced = sigmas
                snr_vals = (1.0 / (sigmas_reduced.to(torch.float32) ** 2)).flatten()
                if (
                    per_sample_vel.numel() > 1
                    and snr_vals.numel() == per_sample_vel.numel()
                ):
                    x = per_sample_vel.to(torch.float32)
                    y = snr_vals
                    x = x - x.mean()
                    y = y - y.mean()
                    # Use population std (correction=0) to avoid degrees-of-freedom issues
                    denom = x.std(correction=0) * y.std(correction=0)
                    corr = (
                        (x * y).mean() / (denom + 1e-12)
                        if torch.isfinite(denom) and float(denom.item()) > 0.0
                        else torch.tensor(0.0, device=x.device)
                    )
                    loss_snr_corr = float(corr.item())
                else:
                    loss_snr_corr = 0.0
            except Exception:
                loss_snr_corr = 0.0

            # Compute simple slope proxy: corr * std(loss) / std(SNR)
            loss_slope = 0.0
            try:
                if sigmas_reduced is not None and per_sample_vel.numel() > 1:
                    snr_vals = (1.0 / (sigmas_reduced.to(torch.float32) ** 2)).flatten()
                    snr_std = snr_vals.std(correction=0)
                    loss_std = per_sample_vel.to(torch.float32).std(correction=0)
                    if (
                        torch.isfinite(snr_std)
                        and float(snr_std.item()) > 0
                        and torch.isfinite(loss_std)
                    ):
                        loss_slope = float(
                            loss_snr_corr * (loss_std / (snr_std + 1e-12))
                        )
            except Exception:
                loss_slope = 0.0

            # SNR namespaces: put essential SNR in snr/ and other SNR in snr_other/
            essential = {
                "train/loss_p50": float(p50),
                "train/loss_p90": float(p90),
                "train/loss_cv_in_batch": float(cv_in_batch),
                "train/direct_noise_loss_mean": float(direct_noise_loss_mean),
                "snr/train/loss_snr_correlation_batch": float(loss_snr_corr),
            }
            # Route slope to snr_other to keep essentials compact
            others = {
                "snr_other/train/loss_snr_slope_batch": float(loss_slope),
            }
            metrics.update(essential)
            metrics.update(others)
        except Exception:
            return {}

        return metrics

    def get_contrastive_flow_components(self) -> Optional[Dict[str, Any]]:
        """
        Get the last computed contrastive flow matching loss components.

        Returns:
            Dictionary with flow_loss, contrastive_loss, lambda_val, and
            used_class_conditioning if contrastive flow matching was used,
            None otherwise.
        """
        return self._last_contrastive_components
