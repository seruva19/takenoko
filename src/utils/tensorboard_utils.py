"""TensorBoard utilities for non-intrusive metric description registration.

If TensorFlow is available, TensorBoard supports per-scalar descriptions that
render directly under charts. This module provides a safe helper to register
those descriptions without impacting the training loop if the environment
doesn't support it.
"""

from __future__ import annotations

from typing import Any, Dict


def get_default_metric_descriptions() -> Dict[str, str]:
    """Return a default set of metric tag descriptions.

    Users can extend or override this mapping before calling
    register_metric_descriptions_non_intrusive.
    """
    return {
        # VAE loss tracking
        "vae/loss/total": "Total VAE training loss after combining all weighted components (lower is better).",
        "vae/loss/average": "Moving average of the VAE training loss (lower is better).",
        "vae/loss/mse": "Mean-squared reconstruction loss term for VAE training (lower is better).",
        "vae/loss/mae": "Mean-absolute reconstruction loss term for VAE training (lower is better).",
        "vae/loss/lpips": "Perceptual LPIPS reconstruction loss for VAE training (lower is better).",
        "vae/loss/edge": "Sobel edge consistency loss for VAE training (lower is better).",
        "vae/loss/kl": "KL divergence term during VAE training (lower is better).",
        "vae/loss/coeff_mse": "Current balancing coefficient applied to the VAE MSE loss (informational).",
        "vae/loss/coeff_mae": "Current balancing coefficient applied to the VAE MAE loss (informational).",
        "vae/loss/coeff_lpips": "Current balancing coefficient applied to the VAE LPIPS loss (informational).",
        "vae/loss/coeff_edge": "Current balancing coefficient applied to the VAE edge loss (informational).",
        "vae/loss/coeff_kl": "Current balancing coefficient applied to the VAE KL loss (informational).",
        "vae/loss/median_mse": "Rolling median magnitude of the VAE MSE loss before balancing (informational).",
        "vae/loss/median_mae": "Rolling median magnitude of the VAE MAE loss before balancing (informational).",
        "vae/loss/median_lpips": "Rolling median magnitude of the VAE LPIPS loss before balancing (informational).",
        "vae/loss/median_edge": "Rolling median magnitude of the VAE edge loss before balancing (informational).",
        "vae/loss/median_kl": "Rolling median magnitude of the VAE KL loss before balancing (informational).",
        "vae/val_loss": "Validation reconstruction loss for the VAE (lower is better).",
        "vae/sample_saved": "Indicator metric marking that VAE samples were saved this step (informational).",
        "vae/sample_lpips": "LPIPS distance between saved VAE samples and reconstructions (lower is better).",
        # Basic loss metrics
        "loss/current": "Current step training loss (lower is better).",
        "loss/average": "Moving average of training loss (lower is better).",
        "loss/ema": "Bias-corrected EMA of loss (lower is better).",
        "loss/mse": "Base flow-matching MSE per step (lower is better).",
        # Training loss components and diagnostics
        "train/loss_p50": "Median per-sample velocity loss (lower is better).",
        "train/loss_p90": "90th percentile per-sample velocity loss (lower is better).",
        "train/loss_cv_in_batch": "Coefficient of variation of per-sample loss within batch; measures training stability (lower is better).",
        "train/direct_noise_loss_mean": "Mean direct noise-prediction loss per batch (lower is better).",
        "train/loss_snr_correlation_batch": "Correlation between loss and SNR within batch; negative values often indicate proper noise scheduling (closer to -1 is better).",
        # Training loss components (optional)
        "train/base_loss": "Base flow-matching loss component before additional terms (lower is better).",
        "train/dop_loss": "Differential Output Preservation loss; maintains base model behavior (lower is better).",
        "train/dispersive_loss": "InfoNCE-style dispersive loss for representation diversity (lower is better).",
        "train/optical_flow_loss": "Temporal consistency loss using optical flow (lower is better).",
        "train/layer_sync_loss": "LayerSync self-alignment loss between transformer blocks (lower is better).",
        "layersync_similarity": "Mean cosine similarity between LayerSync source/target blocks (higher is better).",
        "layersync_loss": "LayerSync self-alignment loss between transformer blocks (lower is better).",
        "train/repa_loss": "REPA alignment loss for visual quality (lower is better).",
        "loss/bfm_semfeat": "BFM SemFeat alignment loss (weighted, lower is better).",
        "bfm_semfeat_similarity": "SemFeat cosine similarity between diffusion and encoder features (higher is better).",
        "loss/bfm_frn": "BFM FRN residual approximation loss (weighted, lower is better).",
        "crepa_loss": "CREPA cross-frame alignment loss for video consistency (lower is better).",
        "crepa_similarity": "Mean CREPA similarity across frames (higher is better).",
        "loss/crepa": "CREPA cross-frame alignment loss for video consistency (lower is better).",
        # Validation metrics - velocity prediction
        "val/velocity_loss_avg": "Average velocity loss across validation timesteps (lower is better).",
        "val/velocity_loss_avg_weighted": "Sample-weighted average velocity loss across all validation data (lower is better).",
        "val/velocity_loss_std": "Standard deviation of velocity loss across timesteps; measures consistency (lower is better).",
        "val/velocity_loss_range": "Range (max-min) of velocity loss across timesteps; measures stability (lower is better).",
        "val/velocity_loss_cv_across_timesteps": "Coefficient of variation of velocity loss across timesteps (lower is better).",
        "val/velocity_loss_avg_p25": "25th percentile of per-timestep velocity loss (lower is better).",
        "val/velocity_loss_avg_p50": "Median of per-timestep velocity loss (lower is better).",
        "val/velocity_loss_avg_p75": "75th percentile of per-timestep velocity loss (lower is better).",
        "val/velocity_loss_snr_correlation": "Correlation between velocity loss and SNR across timesteps; indicates noise scheduling effectiveness (negative preferred).",
        "val/velocity_loss_timestep_correlation": "Correlation between velocity loss and timestep values; shows temporal consistency (closer to 0 is better).",
        "val/best_velocity_loss": "Best (lowest) velocity loss among all validation timesteps (lower is better).",
        "val/worst_velocity_loss": "Worst (highest) velocity loss among all validation timesteps (lower is better).",
        "val/best_timestep_velocity": "Timestep with the best velocity loss performance (informational).",
        "val/worst_timestep_velocity": "Timestep with the worst velocity loss performance (informational).",
        # Validation metrics - direct noise prediction
        "val/direct_noise_loss_avg": "Average direct noise-prediction loss across validation timesteps (lower is better).",
        "val/direct_noise_loss_avg_weighted": "Sample-weighted average direct noise loss across all validation data (lower is better).",
        "val/direct_noise_loss_std": "Standard deviation of direct noise loss across timesteps (lower is better).",
        "val/direct_noise_loss_range": "Range (max-min) of direct noise loss across timesteps (lower is better).",
        "val/direct_noise_loss_cv_across_timesteps": "Coefficient of variation of direct noise loss across timesteps (lower is better).",
        "val/direct_noise_loss_avg_p25": "25th percentile of per-timestep direct noise loss (lower is better).",
        "val/direct_noise_loss_avg_p50": "Median of per-timestep direct noise loss (lower is better).",
        "val/direct_noise_loss_avg_p75": "75th percentile of per-timestep direct noise loss (lower is better).",
        "val/noise_loss_snr_correlation": "Correlation between direct noise loss and SNR across timesteps (negative preferred).",
        "val/noise_loss_timestep_correlation": "Correlation between direct noise loss and timestep values (closer to 0 is better).",
        "val/best_direct_loss": "Best (lowest) direct noise loss among all validation timesteps (lower is better).",
        "val/worst_direct_loss": "Worst (highest) direct noise loss among all validation timesteps (lower is better).",
        "val/best_timestep_direct": "Timestep with the best direct noise loss performance (informational).",
        "val/worst_timestep_direct": "Timestep with the worst direct noise loss performance (informational).",
        # Validation comparison metrics
        "val/loss_ratio": "Ratio of velocity loss to direct noise loss; indicates relative performance (closer to 1.0 is better).",
        # Per-timestep validation metrics (dynamic based on timesteps)
        # Note: These are generated dynamically in validation code as:
        # f"val_timesteps/velocity_loss_mean_t{timestep}"
        # f"val_timesteps/velocity_loss_std_t{timestep}"
        # f"val_timesteps/velocity_loss_min_t{timestep}"
        # f"val_timesteps/velocity_loss_max_t{timestep}"
        # f"val_timesteps/velocity_loss_p50_t{timestep}"
        # f"val_timesteps/velocity_loss_p90_t{timestep}"
        # f"val_timesteps/direct_noise_loss_mean_t{timestep}"
        # f"val_timesteps/direct_noise_loss_std_t{timestep}"
        # f"val_timesteps/direct_noise_loss_min_t{timestep}"
        # f"val_timesteps/direct_noise_loss_max_t{timestep}"
        # f"val_timesteps/direct_noise_loss_p50_t{timestep}"
        # f"val_timesteps/direct_noise_loss_p90_t{timestep}"
        # f"val_timesteps/snr_t{timestep}"
        # Generic timestep patterns for common validation timesteps
        "val_timesteps/velocity_loss_mean": "Mean velocity loss at specific timestep (lower is better).",
        "val_timesteps/velocity_loss_std": "Standard deviation of velocity loss at specific timestep (lower is better).",
        "val_timesteps/velocity_loss_min": "Minimum velocity loss at specific timestep (lower is better).",
        "val_timesteps/velocity_loss_max": "Maximum velocity loss at specific timestep (lower is better).",
        "val_timesteps/velocity_loss_p50": "Median velocity loss at specific timestep (lower is better).",
        "val_timesteps/velocity_loss_p90": "90th percentile velocity loss at specific timestep (lower is better).",
        "val_timesteps/direct_noise_loss_mean": "Mean direct noise loss at specific timestep (lower is better).",
        "val_timesteps/direct_noise_loss_std": "Standard deviation of direct noise loss at specific timestep (lower is better).",
        "val_timesteps/direct_noise_loss_min": "Minimum direct noise loss at specific timestep (lower is better).",
        "val_timesteps/direct_noise_loss_max": "Maximum direct noise loss at specific timestep (lower is better).",
        "val_timesteps/direct_noise_loss_p50": "Median direct noise loss at specific timestep (lower is better).",
        "val_timesteps/direct_noise_loss_p90": "90th percentile direct noise loss at specific timestep (lower is better).",
        "val_timesteps/snr": "Signal-to-Noise Ratio at specific timestep; higher values indicate less noise (higher is better).",
        # Per-source loss metrics (video vs image)
        "loss/video": "Training loss for video data sources only (lower is better).",
        "loss/image": "Training loss for image data sources only (lower is better).",
        "loss/source": "Training loss for specific data source type (lower is better).",
        # REPA metrics (Representation Alignment)
        "repa/total_loss": "Total REPA alignment loss across all encoders and layers (lower is better).",
        "repa/loss_lambda": "REPA loss weighting factor in total loss computation (informational).",
        "repa/num_encoders": "Number of active encoders in REPA ensemble (informational).",
        "repa/num_alignment_layers": "Number of diffusion transformer layers used for alignment (informational).",
        # REPA per-encoder metrics
        "repa/encoders/*/feature_norm": "L2 norm of features from specific encoder (stable values preferred).",
        "repa/encoders/*/feature_mean": "Mean value of features from specific encoder (stable values preferred).",
        "repa/encoders/*/feature_std": "Standard deviation of features from specific encoder (stable values preferred).",
        "repa/encoders/*/layer_count": "Number of alignment layers for specific encoder (informational).",
        # REPA per-layer metrics
        "repa/layers/*/similarity_mean": "Mean cosine similarity between diffusion and encoder features at specific layer (higher values indicate better alignment).",
        "repa/layers/*/similarity_std": "Standard deviation of cosine similarities at specific layer (lower values indicate more consistent alignment).",
        "repa/layers/*/similarity_min": "Minimum cosine similarity at specific layer (higher values indicate worst-case alignment quality).",
        "repa/layers/*/similarity_max": "Maximum cosine similarity at specific layer (informational).",
        "repa/layers/*/loss_mean": "Mean alignment loss (negative cosine similarity) at specific layer (lower is better).",
        "repa/layers/*/loss_std": "Standard deviation of alignment loss at specific layer (lower values indicate more stable alignment).",
        "repa/layers/*/loss_min": "Minimum alignment loss at specific layer (lower is better).",
        "repa/layers/*/loss_max": "Maximum alignment loss at specific layer (informational).",
        # REPA summary statistics
        "repa/summary/similarity_mean": "Mean cosine similarity across all encoders and layers (higher values indicate better overall alignment).",
        "repa/summary/similarity_std": "Standard deviation of cosine similarities across all encoders and layers (lower values indicate more consistent alignment).",
        "repa/summary/similarity_min": "Minimum cosine similarity across all encoders and layers (higher values indicate worst-case alignment quality).",
        "repa/summary/similarity_max": "Maximum cosine similarity across all encoders and layers (informational).",
        "repa/summary/loss_mean": "Mean alignment loss across all encoders and layers (lower is better).",
        "repa/summary/loss_std": "Standard deviation of alignment loss across all encoders and layers (lower values indicate more stable alignment).",
        "repa/summary/loss_min": "Minimum alignment loss across all encoders and layers (lower is better).",
        "repa/summary/loss_max": "Maximum alignment loss across all encoders and layers (informational).",
        # Gradient monitoring
        "grad_norm": "Global gradient norm across all model parameters; indicates gradient flow strength (moderate values preferred).",
        # Learning rate tracking
        "lr/unet": "Learning rate for UNet/main model parameters (informational).",
        "lr/textencoder": "Learning rate for text encoder parameters (informational).",
        "lr/group0": "Learning rate for parameter group 0 (informational).",
        "lr/group1": "Learning rate for parameter group 1 (informational).",
        "lr/d*lr": "Effective learning rate (d*lr) for adaptive optimizers like Prodigy (informational).",
        "lr/d*lr/unet": "Effective learning rate for UNet parameters with adaptive optimizers (informational).",
        "lr/d*lr/textencoder": "Effective learning rate for text encoder with adaptive optimizers (informational).",
        "lr/d*lr/group0": "Effective learning rate for parameter group 0 with adaptive optimizers (informational).",
        "lr/d*lr/group1": "Effective learning rate for parameter group 1 with adaptive optimizers (informational).",
        # Parameter statistics (when log_param_stats=True)
        "param_stats/total_param_norm": "Sum of norms of all model parameters; tracks overall parameter scale (stable values preferred).",
        "param_stats/avg_param_norm": "Average norm across all model parameters (stable values preferred).",
        "param_stats/total_grad_norm": "Sum of gradient norms across all parameters; indicates total gradient magnitude (moderate values preferred).",
        "param_stats/avg_grad_norm": "Average gradient norm across all parameters (moderate values preferred).",
        "param_stats/num_params": "Number of trainable parameters (informational).",
        "param_stats/largest_param_norm": "Largest parameter norm in the model; helps detect parameter explosion (stable values preferred).",
        "param_stats/largest_grad_norm": "Largest gradient norm in the model; helps detect gradient explosion (moderate values preferred).",
        # Individual parameter tracking (top-K parameters by norm)
        "param_norm": "L2 norm of specific parameter tensor; tracks parameter drift (stable values preferred).",
        "grad_norm": "L2 norm of gradients for specific parameter tensor (moderate values preferred).",
    }


def register_metric_descriptions_non_intrusive(
    accelerator: Any, args: Any, tag_to_desc: Dict[str, str]
) -> None:
    """Attempt to register TensorBoard scalar descriptions; fail silently.

    - No-ops unless logging with TensorBoard (args.log_with in ["tensorboard", "all"]).
    - Requires TensorFlow to be importable; otherwise returns quietly.
    - Writes a single scalar per tag with a description at step=0 so that the
      description is rendered under the corresponding chart in TensorBoard.
    """
    try:
        if getattr(args, "log_with", None) not in ["tensorboard", "all"]:
            return

        # Find TensorBoard writer log_dir
        log_dir = None
        try:
            for t in getattr(accelerator, "trackers", []):
                if getattr(t, "name", "") == "tensorboard" and hasattr(
                    t.writer, "log_dir"
                ):
                    log_dir = t.writer.log_dir
                    break
        except Exception:
            log_dir = None
        if log_dir is None:
            return

        try:
            import tensorflow as tf  # type: ignore
        except Exception:
            # TensorFlow not available: cannot attach per-scalar descriptions
            return

        try:
            writer = tf.summary.create_file_writer(log_dir)
            with writer.as_default():
                for tag, desc in tag_to_desc.items():
                    try:
                        tf.summary.scalar(name=tag, data=0.0, step=0, description=desc)
                    except Exception:
                        # Skip tags that fail; continue registering others
                        pass
                writer.flush()
        except Exception:
            # Do not raise; remain non-intrusive
            return
    except Exception:
        # Fully non-intrusive
        return


def _infer_direction_hint(tag: str) -> str | None:
    """Infer whether higher or lower is better for a given metric tag.

    Returns one of: "down", "up", or None when unknown/neutral.
    Heuristics are intentionally simple to avoid over-engineering.
    """
    t = tag.lower()

    # Do not decorate clearly informational tags
    neutral_keywords = [
        "lr/",  # learning rates
        "/snr",  # signal-to-noise ratios (context dependent)
        "snr/",
        "param_stats/",
        "val_timesteps/snr",
    ]
    if any(k in t for k in neutral_keywords):
        return None

    lower_is_better = [
        "loss",
        "error",
        "mse",
        "mae",
        "rmse",
        "nll",
        "perplex",
        "grad",
        "gradient",
        "norm",
        "wd",
        "weight_decay",
        "latency",
        "time",
        "duration",
        "memory",
        "mem",
    ]
    if any(k in t for k in lower_is_better):
        return "down"

    higher_is_better = [
        "accuracy",
        "acc/",
        "/acc",
        "f1",
        "precision",
        "recall",
        "psnr",
        "ssim",
        "iou",
        "bleu",
        "rouge",
        "throughput",
        "speed",
        "samples_per_sec",
        "tokens_per_sec",
        "ips",
        "itps",
        "fps",
        "sps",
    ]
    if any(k in t for k in higher_is_better):
        return "up"

    return None


def apply_direction_hints_to_logs(args: Any, logs: Dict[str, Any]) -> Dict[str, Any]:
    """Append small emoji hints to TensorBoard metric tags when enabled.

    - Controlled via args.tensorboard_append_direction_hints (bool).
    - Example: "loss/current" -> "loss/current (📉 better)"; "throughput/sps" -> "throughput/sps (📈 better)"
    - Avoids modifying tags that already contain emoji hints.
    - Leaves unrelated entries (non-scalar values) untouched key-wise.
    """
    try:
        if not bool(getattr(args, "tensorboard_append_direction_hints", False)):
            return logs

        new_logs: Dict[str, Any] = {}
        for k, v in logs.items():
            try:
                # Only decorate string tags
                if not isinstance(k, str):
                    new_logs[k] = v
                    continue

                # Skip if already has a direction emoji
                stripped_key = k.split("/")[-1]
                if "📉" in stripped_key or "📈" in stripped_key:
                    new_logs[k] = v
                    continue

                hint = _infer_direction_hint(k)
                if hint == "down":
                    decorated_leaf = f"{stripped_key} (📉 better)"
                elif hint == "up":
                    decorated_leaf = f"{stripped_key} (📈 better)"
                else:
                    new_logs[k] = v
                    continue

                if "/" in k:
                    parts = k.split("/")
                    parts[-1] = decorated_leaf
                    new_key = "/".join(parts)
                else:
                    new_key = decorated_leaf

                new_logs[new_key] = v
            except Exception:
                new_logs[k] = v

        return new_logs
    except Exception:
        return logs
