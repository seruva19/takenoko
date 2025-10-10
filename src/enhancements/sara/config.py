"""SARA configuration management."""

from dataclasses import dataclass
from typing import Optional

from common.logger import get_logger


logger = get_logger(__name__)


@dataclass
class SaraConfig:
    """Typed container for SARA-specific hyperparameters."""

    enabled: bool = False

    encoder_name: str = "dinov2_vitb14"
    encoder_cache_dir: str = "models"
    alignment_depth: int = 8

    patch_loss_weight: float = 0.5
    autocorr_loss_weight: float = 0.5
    adversarial_loss_weight: float = 0.05

    autocorr_normalize: bool = True
    autocorr_use_frobenius: bool = True

    adversarial_enabled: bool = True
    discriminator_arch: str = "resnet18"
    discriminator_lr: float = 2e-4
    discriminator_updates_per_step: int = 1
    discriminator_warmup_steps: int = 500
    discriminator_update_interval: int = 5

    similarity_fn: str = "cosine"
    gradient_penalty_weight: float = 0.0
    feature_matching: bool = False
    feature_matching_weight: float = 0.1

    cache_encoder_outputs: bool = True
    use_mixed_precision: bool = True
    discriminator_max_grad_norm: Optional[float] = None
    discriminator_scheduler_step: int = 0
    discriminator_scheduler_gamma: float = 0.1
    log_detailed_metrics: bool = False

    @classmethod
    def from_args(cls, args: any) -> Optional["SaraConfig"]:
        """Build a config from an argparse namespace.

        Note: All SARA parameters are now parsed in config_parser.py for consistency
        with other features like REPA. This method just constructs the config object
        from the already-parsed args attributes.
        """
        if not args.sara_enabled:
            return None

        config = cls(
            enabled=args.sara_enabled,
            encoder_name=args.sara_encoder_name,
            encoder_cache_dir=getattr(args, "model_cache_dir", "models"),
            alignment_depth=args.sara_alignment_depth,
            patch_loss_weight=args.sara_patch_loss_weight,
            autocorr_loss_weight=args.sara_autocorr_loss_weight,
            adversarial_loss_weight=args.sara_adversarial_loss_weight,
            autocorr_normalize=args.sara_autocorr_normalize,
            autocorr_use_frobenius=args.sara_autocorr_use_frobenius,
            adversarial_enabled=args.sara_adversarial_enabled,
            discriminator_arch=args.sara_discriminator_arch,
            discriminator_lr=args.sara_discriminator_lr,
            discriminator_updates_per_step=args.sara_discriminator_updates_per_step,
            discriminator_warmup_steps=args.sara_discriminator_warmup_steps,
            discriminator_update_interval=args.sara_discriminator_update_interval,
            similarity_fn=args.sara_similarity_fn,
            gradient_penalty_weight=args.sara_gradient_penalty_weight,
            feature_matching=args.sara_feature_matching,
            feature_matching_weight=args.sara_feature_matching_weight,
            cache_encoder_outputs=args.sara_cache_encoder_outputs,
            use_mixed_precision=args.sara_use_mixed_precision,
            discriminator_max_grad_norm=getattr(
                args, "sara_discriminator_max_grad_norm", None
            ),
            discriminator_scheduler_step=getattr(
                args, "sara_discriminator_scheduler_step", 0
            ),
            discriminator_scheduler_gamma=getattr(
                args, "sara_discriminator_scheduler_gamma", 0.1
            ),
            log_detailed_metrics=getattr(
                args, "sara_log_detailed_metrics", False
            ),
        )
        config.validate()
        logger.info("SARA configuration created: %s", config)
        return config

    def validate(self) -> None:
        """Check for obviously invalid hyperparameters."""
        if self.patch_loss_weight < 0:
            raise ValueError(
                f"patch_loss_weight must be >= 0, got {self.patch_loss_weight}"
            )
        if self.patch_loss_weight == 0 and self.autocorr_loss_weight == 0:
            if not self.adversarial_enabled or self.adversarial_loss_weight <= 0:
                raise ValueError(
                    "At least one SARA loss component must be active; "
                    "all weights are zero or disabled."
                )
        if self.autocorr_loss_weight < 0:
            raise ValueError(
                f"autocorr_loss_weight must be >= 0, got {self.autocorr_loss_weight}"
            )
        if self.adversarial_loss_weight < 0:
            raise ValueError(
                f"adversarial_loss_weight must be >= 0, got {self.adversarial_loss_weight}"
            )
        if self.adversarial_enabled and self.adversarial_loss_weight == 0:
            raise ValueError(
                "adversarial_loss_weight must be > 0 when adversarial alignment is enabled"
            )
        if not self.adversarial_enabled and self.adversarial_loss_weight > 0:
            logger.warning(
                "SARA adversarial alignment disabled but adversarial_loss_weight > 0. "
                "The weight will be ignored."
            )
        if self.alignment_depth < 1:
            raise ValueError(
                f"alignment_depth must be >= 1, got {self.alignment_depth}"
            )
        if self.discriminator_lr <= 0:
            raise ValueError(
                f"discriminator_lr must be > 0, got {self.discriminator_lr}"
            )
        if self.discriminator_updates_per_step < 1:
            raise ValueError(
                "discriminator_updates_per_step must be >= 1, "
                f"got {self.discriminator_updates_per_step}"
            )
        if self.discriminator_warmup_steps < 0:
            raise ValueError(
                f"discriminator_warmup_steps must be >= 0, got {self.discriminator_warmup_steps}"
            )
        if self.discriminator_update_interval < 1:
            raise ValueError(
                "discriminator_update_interval must be >= 1, "
                f"got {self.discriminator_update_interval}"
            )
        if self.similarity_fn not in {"cosine", "mse", "l1"}:
            raise ValueError(
                f"similarity_fn must be one of {{'cosine','mse','l1'}}, got {self.similarity_fn}"
            )
        if self.gradient_penalty_weight < 0:
            raise ValueError(
                f"gradient_penalty_weight must be >= 0, got {self.gradient_penalty_weight}"
            )
        if self.discriminator_max_grad_norm is not None and self.discriminator_max_grad_norm <= 0:
            raise ValueError(
                "discriminator_max_grad_norm must be > 0 when provided"
            )
        if self.discriminator_scheduler_step < 0:
            raise ValueError(
                "discriminator_scheduler_step must be >= 0"
            )
        if self.discriminator_scheduler_gamma <= 0:
            raise ValueError("discriminator_scheduler_gamma must be > 0")
        if self.feature_matching_weight < 0:
            raise ValueError("feature_matching_weight must be >= 0")
