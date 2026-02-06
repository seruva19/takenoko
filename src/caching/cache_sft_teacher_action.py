import argparse

from common.logger import get_logger
from dataset import config_utils
from dataset.config_utils import BlueprintGenerator, ConfigSanitizer

from caching.cache_sft_teacher_features import cache_sft_teacher_features

logger = get_logger(__name__)


def build_sft_cache_precompute_args(
    args: argparse.Namespace,
    overwrite_existing: bool | None = None,
) -> argparse.Namespace:
    """Create cache-precompute args with explicit skip/overwrite behavior."""
    cache_args = argparse.Namespace(**vars(args))
    cache_args.dataset_config = args.dataset_config
    cache_args.device = getattr(args, "device", None)

    overwrite = bool(
        getattr(cache_args, "sft_teacher_cache_overwrite_existing", False)
        if overwrite_existing is None
        else overwrite_existing
    )
    if overwrite:
        cache_args.sft_teacher_cache_skip_existing = False
        cache_args.sft_teacher_cache_purge = True
        cache_args.sft_teacher_cache_mode = "write"
    else:
        cache_args.sft_teacher_cache_skip_existing = True
        if str(getattr(cache_args, "sft_teacher_cache_mode", "off")).lower() == "off":
            cache_args.sft_teacher_cache_mode = "read_write"
    return cache_args


def run_cache_sft_teacher_action(args: argparse.Namespace) -> bool:
    """Run offline SFT teacher-feature cache precompute operation."""
    logger.info("Starting SFT teacher-feature caching...")

    try:
        cache_args = build_sft_cache_precompute_args(args)

        if not bool(getattr(cache_args, "enable_structure_from_tracking", False)):
            logger.warning(
                "enable_structure_from_tracking is false; proceeding with SFT cache precompute using sft_* settings only."
            )

        if str(getattr(cache_args, "sft_teacher_cache_dir", "")).strip() == "":
            raise ValueError(
                "sft_teacher_cache_dir must be set to run cache_sft_teacher_features."
            )

        blueprint_generator = BlueprintGenerator(ConfigSanitizer())
        logger.info(f"Load dataset config from {cache_args.dataset_config}")
        user_config = config_utils.load_user_config(cache_args.dataset_config)
        blueprint = blueprint_generator.generate(user_config, cache_args)

        all_dataset_blueprints = list(blueprint.train_dataset_group.datasets)
        if len(blueprint.val_dataset_group.datasets) > 0:
            all_dataset_blueprints.extend(blueprint.val_dataset_group.datasets)
        combined_dataset_group_blueprint = config_utils.DatasetGroupBlueprint(
            all_dataset_blueprints
        )
        dataset_group = config_utils.generate_dataset_group_by_blueprint(
            combined_dataset_group_blueprint,
            training=False,
            prior_loss_weight=getattr(cache_args, "prior_loss_weight", 1.0),
        )

        datasets = dataset_group.datasets
        cache_sft_teacher_features(datasets, cache_args)  # type: ignore[arg-type]
        logger.info("SFT teacher-feature caching completed successfully!")
        return True
    except Exception as exc:
        logger.exception(f"Error during SFT teacher-feature caching: {exc}")
        return False
