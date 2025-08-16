from __future__ import annotations

import os
from typing import Any

from common.logger import get_logger
from dataset import config_utils

logger = get_logger(__name__)


def maybe_wrap_with_self_correction(
    args: Any,
    blueprint_generator: Any,
    user_config: dict,
    model_config: Any,
    train_dataset_group: Any,
):
    """Optionally wrap the main dataset group with HybridDatasetGroup mixing a correction cache dataset.

    Returns the (possibly wrapped) dataset group.
    """
    try:
        from dataset.hybrid_group import HybridDatasetGroup  # type: ignore

        use_self_correction = bool(getattr(args, "self_correction_enabled", False))
        if not use_self_correction:
            return train_dataset_group

        correction_dir = os.path.join(args.output_dir, "self_correction_cache")
        if not os.path.exists(correction_dir):
            logger.info(
                "Self-correction enabled but no cache directory exists yet; training with main dataset only"
            )
            return train_dataset_group

        # Defaults if config doesn't expose fields
        default_w, default_h, default_frames = 512, 512, 17
        try:
            if hasattr(model_config, "train_img_size"):
                default_w = int(model_config.train_img_size[0])  # type: ignore
                default_h = int(model_config.train_img_size[1])  # type: ignore
            if hasattr(model_config, "train_img_frames"):
                default_frames = int(model_config.train_img_frames)  # type: ignore
        except Exception:
            pass

        correction_cfg = {
            "datasets": {
                "train": [
                    {
                        "video_directory": correction_dir,
                        "resolution": [default_w, default_h],
                        "batch_size": 1,
                        "num_repeats": 1,
                        "enable_bucket": False,
                        "bucket_no_upscale": False,
                        "target_frames": [default_frames],
                    }
                ]
            }
        }
        corr_blueprint = blueprint_generator.generate(correction_cfg, args)
        correction_group = config_utils.generate_dataset_group_by_blueprint(
            corr_blueprint.train_dataset_group,
            training=True,
            enable_control_lora=getattr(args, "enable_control_lora", False),
            prior_loss_weight=getattr(args, "prior_loss_weight", 1.0),
        )
        ratio = float(getattr(args, "self_correction_batch_ratio", 0.2))
        wrapped = HybridDatasetGroup(train_dataset_group, correction_group, ratio)
        logger.info(
            f"Self-correction enabled: mixing correction dataset at ratio={ratio}"
        )
        return wrapped
    except Exception as _sc_wrap_err:  # noqa: BLE001
        logger.warning(f"Self-correction hybrid setup skipped: {_sc_wrap_err}")
        return train_dataset_group


def maybe_attach_self_correction_manager(
    args: Any,
    accelerator: Any,
    sampling_manager: Any,
    blueprint: Any,
    vae_dtype: Any,
    transformer: Any,
):
    """Attach a SelfCorrectionManager instance to accelerator/model for periodic refresh, if enabled."""
    if not bool(getattr(args, "self_correction_enabled", False)):
        return
    if sampling_manager is None:
        return

    from self_correction.manager import SelfCorrectionManager

    sc_manager = SelfCorrectionManager(
        args=args,
        accelerator=accelerator,
        sampling_manager=sampling_manager,
        blueprint=blueprint,
        vae_dtype=vae_dtype,
    )
    try:
        setattr(accelerator.state, "_self_correction_manager", sc_manager)
    except Exception:
        pass
    try:
        setattr(transformer, "_self_correction_manager", sc_manager)
    except Exception:
        pass
