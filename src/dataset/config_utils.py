## Based on: https://github.com/kohya-ss/musubi-tuner/blob/main/src/musubi_tuner/dataset/config_utils.py (Apache)

import argparse
from dataclasses import (
    asdict,
    dataclass,
)
import functools
import logging
import random
from textwrap import dedent
from pathlib import Path

# from toolz import curry
from typing import List, Optional, Sequence, Tuple, Union

import toml
import voluptuous
from voluptuous import (
    Any,
    ExactSequence,
    MultipleInvalid,
    Object,
    Schema,
    Optional as VOptional,
)

from dataset.image_video_dataset import DatasetGroup, ImageDataset, VideoDataset

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


@dataclass
class BaseDatasetParams:
    resolution: Tuple[int, int] = (960, 544)
    enable_bucket: bool = False
    bucket_no_upscale: bool = False
    caption_extension: Optional[str] = None
    caption_dropout_rate: float = 0.0
    batch_size: int = 1
    num_repeats: int = 1
    cache_directory: Optional[str] = None
    debug_dataset: bool = False  # Fixed: was tuple, now proper bool
    is_val: bool = False  # Fixed: was tuple, now proper bool
    target_model: Optional[str] = None  # Model type for cache postfix determination
    is_reg: bool = False  # for regularization datasets
    mask_path: Optional[str] = None  # Path to mask images/videos for masked training


@dataclass
class ImageDatasetParams(BaseDatasetParams):
    image_directory: Optional[str] = None


@dataclass
class VideoDatasetParams(BaseDatasetParams):
    video_directory: Optional[str] = None
    target_frames: Sequence[int] = (1,)
    frame_extraction: Optional[str] = "head"
    frame_stride: Optional[int] = 1
    frame_sample: Optional[int] = 1
    max_frames: Optional[int] = 129
    source_fps: Optional[float] = None


@dataclass
class DatasetBlueprint:
    is_image_dataset: bool
    params: Union[ImageDatasetParams, VideoDatasetParams]


@dataclass
class DatasetGroupBlueprint:
    datasets: Sequence[DatasetBlueprint]


@dataclass
class Blueprint:
    # dataset_group: DatasetGroupBlueprint
    train_dataset_group: DatasetGroupBlueprint
    val_dataset_group: DatasetGroupBlueprint


class ConfigSanitizer:
    # @curry  # TODO: Commented out decorator - consider if curry is needed or remove comment
    @staticmethod
    def __validate_and_convert_twodim(klass, value: Sequence) -> Tuple:
        Schema(ExactSequence([klass, klass]))(value)
        return tuple(value)

    # @curry  # TODO: Commented out decorator - consider if curry is needed or remove comment
    @staticmethod
    def __validate_and_convert_scalar_or_twodim(
        klass, value: Union[float, Sequence]
    ) -> Tuple:
        Schema(Any(klass, ExactSequence([klass, klass])))(value)
        try:
            # First check if it's a scalar value
            Schema(klass)(value)
            return (value, value)
        except (MultipleInvalid, ValueError):
            # If scalar validation fails, it must be a sequence, so validate as twodim
            if isinstance(value, (list, tuple)):
                return ConfigSanitizer.__validate_and_convert_twodim(klass, value)
            else:
                # If it's not a scalar that matches klass and not a sequence, re-raise
                raise

    # datasets schema
    DATASET_ASCENDABLE_SCHEMA = {
        "caption_extension": str,
        "caption_dropout_rate": float,
        "batch_size": int,
        "num_repeats": int,
        "resolution": functools.partial(
            __validate_and_convert_scalar_or_twodim.__func__, int
        ),
        "enable_bucket": bool,
        "bucket_no_upscale": bool,
        "mask_path": str,
    }
    IMAGE_DATASET_DISTINCT_SCHEMA = {
        "image_directory": str,
        "cache_directory": str,
    }
    VIDEO_DATASET_DISTINCT_SCHEMA = {
        "video_directory": str,
        "target_frames": [int],
        "frame_extraction": str,
        "frame_stride": int,
        "frame_sample": int,
        "max_frames": int,
        "cache_directory": str,
        "source_fps": float,
    }

    # options handled by argparse but not handled by user config
    ARGPARSE_SPECIFIC_SCHEMA = {
        "debug_dataset": bool,
        "target_model": str,  # Model type for cache postfix determination
    }

    def __init__(self) -> None:
        self.image_dataset_schema = self.__merge_dict(
            self.DATASET_ASCENDABLE_SCHEMA,
            self.IMAGE_DATASET_DISTINCT_SCHEMA,
        )
        self.video_dataset_schema = self.__merge_dict(
            self.DATASET_ASCENDABLE_SCHEMA,
            self.VIDEO_DATASET_DISTINCT_SCHEMA,
        )

        def validate_flex_dataset(dataset_config: dict):
            if "video_directory" in dataset_config:
                return Schema(self.video_dataset_schema)(dataset_config)
            else:
                return Schema(self.image_dataset_schema)(dataset_config)

        # REFACTOR: Consider extracting this to a separate method for better testability and readability

        self.dataset_schema = validate_flex_dataset

        self.general_schema = self.__merge_dict(
            self.DATASET_ASCENDABLE_SCHEMA,
        )

        nested_or_flat_datasets_schema = Any(
            {
                "general": self.general_schema,
                "train": VOptional([self.dataset_schema]),
                "val": VOptional([self.dataset_schema]),
            },
            [self.dataset_schema],  # legacy flat list of datasets
        )

        # Allow all keys to pass through; rely on downstream processing for detailed checks
        self.user_config_validator = Schema({}, extra=voluptuous.ALLOW_EXTRA)

        # TODO: Large block of commented out code - consider removing if no longer needed
        # self.user_config_validator = Schema(
        #     {
        #         "general": self.general_schema,
        #         "datasets": [self.dataset_schema],
        #     }
        # )

        self.argparse_schema = self.__merge_dict(
            self.ARGPARSE_SPECIFIC_SCHEMA,
        )
        self.argparse_config_validator = Schema(
            Object(self.argparse_schema), extra=voluptuous.ALLOW_EXTRA
        )

    def sanitize_user_config(self, user_config: dict) -> dict:
        try:
            sanitized = self.user_config_validator(user_config)
            logger.info(f"Sanitized config keys: {list(sanitized.keys())}")
            if "datasets" in sanitized:
                datasets_section = sanitized["datasets"]
                if isinstance(datasets_section, list):
                    # Flat list style: datasets is a list of dataset configs
                    logger.info(
                        f"Datasets structure: flat list with {len(datasets_section)} items"
                    )
                    logger.info(f"Train datasets count: {len(datasets_section)}")
                else:
                    # Nested dictionary style: datasets contains train/val/general sections
                    logger.info(f"Datasets keys: {list(datasets_section.keys())}")
                    logger.info(
                        f"Train datasets count: {len(datasets_section.get('train', []))}"
                    )
            return sanitized
        except MultipleInvalid:
            # TODO: clarify the error message - should provide more specific error details
            logger.error("Invalid user config")
            raise

    # NOTE: In nature, argument parser result is not needed to be sanitize
    #   However this will help us to detect program bug
    # TODO: Consider if this sanitization is still necessary or if it adds unnecessary overhead
    def sanitize_argparse_namespace(
        self, argparse_namespace: argparse.Namespace
    ) -> argparse.Namespace:
        try:
            return self.argparse_config_validator(argparse_namespace)
        except MultipleInvalid:
            # XXX: this should be a bug
            logger.error(
                "Invalid cmdline parsed arguments. This should be a bug."
            )  # TODO: XXX comment suggests this is a critical issue - investigate
            raise

    # NOTE: value would be overwritten by latter dict if there is already the same key
    @staticmethod
    def __merge_dict(*dict_list: dict) -> dict:
        merged = {}
        for schema in dict_list:
            # merged |= schema  # TODO: Commented out modern dict union operator - consider using it
            for k, v in schema.items():
                merged[k] = v
        return merged


class BlueprintGenerator:
    BLUEPRINT_PARAM_NAME_TO_CONFIG_OPTNAME = (
        {}
    )  # TODO: Empty dict - consider if this should be populated or removed - REFACTOR: Either populate this mapping or remove if unused

    def __init__(self, sanitizer: ConfigSanitizer):
        self.sanitizer = sanitizer

    # runtime_params is for parameters which is only configurable on runtime, such as tokenizer
    def generate(
        self,
        user_config: dict,
        argparse_namespace: argparse.Namespace,
        **runtime_params,
    ) -> Blueprint:
        sanitized_user_config = self.sanitizer.sanitize_user_config(user_config)
        sanitized_argparse_namespace = self.sanitizer.sanitize_argparse_namespace(
            argparse_namespace
        )

        argparse_config = {
            k: v for k, v in vars(sanitized_argparse_namespace).items() if v is not None
        }
        datasets_section = sanitized_user_config.get("datasets", {})
        # Determine general, train, val depending on structure
        if isinstance(datasets_section, list):
            # Flat list style: everything in datasets is training list
            general_config = sanitized_user_config.get("general", {})
            train_dataset_configs = datasets_section
            val_dataset_configs = sanitized_user_config.get("val_datasets", [])
        else:
            general_config = datasets_section.get("general", {})
            train_dataset_configs = datasets_section.get("train", [])
            val_dataset_configs = datasets_section.get("val", [])

        logger.info(f"Found {len(train_dataset_configs)} training dataset configs")
        train_blueprints = []
        for i, dataset_config in enumerate(train_dataset_configs):
            logger.info(f"Processing training dataset {i}: {dataset_config}")
            is_image_dataset = "image_directory" in dataset_config
            dataset_params_klass = (
                ImageDatasetParams if is_image_dataset else VideoDatasetParams
            )
            params = self.generate_params_by_fallbacks(
                dataset_params_klass,
                [dataset_config, general_config, argparse_config, runtime_params],
            )
            # is_val defaults to False in the dataclass so nothing special is needed
            train_blueprints.append(DatasetBlueprint(is_image_dataset, params))

        # Process validation datasets: mark them as validation.
        logger.info(f"Found {len(val_dataset_configs)} validation dataset configs")
        val_blueprints = []
        for i, dataset_config in enumerate(val_dataset_configs):
            logger.info(f"Processing validation dataset {i}: {dataset_config}")
            is_image_dataset = "image_directory" in dataset_config
            dataset_params_klass = (
                ImageDatasetParams if is_image_dataset else VideoDatasetParams
            )
            params = self.generate_params_by_fallbacks(
                dataset_params_klass,
                [dataset_config, general_config, argparse_config, runtime_params],
            )
            params.is_val = True  # mark as validation
            val_blueprints.append(DatasetBlueprint(is_image_dataset, params))

        train_dataset_group_blueprint = DatasetGroupBlueprint(train_blueprints)
        val_dataset_group_blueprint = DatasetGroupBlueprint(val_blueprints)

        logger.info(
            f"Created {len(train_blueprints)} training blueprints and {len(val_blueprints)} validation blueprints"
        )

        return Blueprint(
            train_dataset_group=train_dataset_group_blueprint,
            val_dataset_group=val_dataset_group_blueprint,
        )

    @staticmethod
    def generate_params_by_fallbacks(param_klass, fallbacks: Sequence[dict]):
        name_map = BlueprintGenerator.BLUEPRINT_PARAM_NAME_TO_CONFIG_OPTNAME
        search_value = BlueprintGenerator.search_value
        default_params = asdict(param_klass())
        param_names = default_params.keys()

        params = {
            name: search_value(
                name_map.get(name, name), fallbacks, default_params.get(name)
            )
            for name in param_names
        }

        return param_klass(**params)

    @staticmethod
    def search_value(
        key: str, fallbacks: Sequence[dict], default_value=None
    ):  # TODO: Missing type hint for default_value - REFACTOR: Add proper type hint (e.g., Optional[Any])
        for cand in fallbacks:
            value = cand.get(key)
            if value is not None:
                return value

        return default_value


def _format_dataset_info_table(
    datasets: List[Union[ImageDataset, VideoDataset]],
) -> str:
    """Format dataset information as a clean table for better readability."""
    if not datasets:
        return "No datasets configured."

    # Define column headers and widths
    headers = [
        "ID",
        "Type",
        "Val",
        "Resolution",
        "Batch",
        "Repeats",
        "Bucket",
        "Directory",
        "Cache Dir",
    ]

    # Calculate column widths based on content
    rows = []
    for i, dataset in enumerate(datasets):
        is_image_dataset = isinstance(dataset, ImageDataset)
        dataset_type = "Image" if is_image_dataset else "Video"
        validation = "Yes" if dataset.is_val else "No"
        resolution = f"{dataset.resolution[0]}x{dataset.resolution[1]}"

        # Get the appropriate directory path
        if is_image_dataset:
            directory = dataset.image_directory or "N/A"
        else:
            directory = dataset.video_directory or "N/A"

        # Truncate long paths for better table display
        if len(directory) > 30:
            directory = "..." + directory[-27:]

        cache_dir = dataset.cache_directory or "N/A"
        if len(cache_dir) > 20:
            cache_dir = "..." + cache_dir[-17:]

        bucket_info = "Yes" if dataset.enable_bucket else "No"
        if dataset.enable_bucket and dataset.bucket_no_upscale:
            bucket_info += "/NoUp"

        row = [
            str(i),
            dataset_type,
            validation,
            resolution,
            str(dataset.batch_size),
            str(dataset.num_repeats),
            bucket_info,
            directory,
            cache_dir,
        ]
        rows.append(row)

    # Calculate column widths
    col_widths = [len(header) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    # Build the table
    table_lines = []

    # Header
    header_line = " | ".join(
        header.ljust(col_widths[i]) for i, header in enumerate(headers)
    )
    table_lines.append(header_line)

    # Separator
    separator = "-+-".join("-" * width for width in col_widths)
    table_lines.append(separator)

    # Data rows
    for row in rows:
        row_line = " | ".join(
            str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)
        )
        table_lines.append(row_line)

    return "\n".join(table_lines)


def _format_dataset_details_table(
    datasets: List[Union[ImageDataset, VideoDataset]],
) -> str:
    """Format detailed dataset information in a consistent table format for all dataset types."""
    if not datasets:
        return "No datasets configured."

    details_lines = []

    for i, dataset in enumerate(datasets):
        is_image_dataset = isinstance(dataset, ImageDataset)
        dataset_type = "Image" if is_image_dataset else "Video"

        # Common details for all datasets
        details_lines.append(
            f"📁 Dataset {i} ({dataset_type}) - Detailed Configuration:"
        )

        # Common parameters
        details_lines.append(f"│ Type                │ {dataset_type}")
        details_lines.append(
            f"│ Validation Dataset  │ {'Yes' if dataset.is_val else 'No'}"
        )
        details_lines.append(
            f"│ Resolution          │ {dataset.resolution[0]} x {dataset.resolution[1]}"
        )
        details_lines.append(f"│ Batch Size          │ {dataset.batch_size}")
        details_lines.append(f"│ Num Repeats         │ {dataset.num_repeats}")
        details_lines.append(
            f"│ Enable Bucket       │ {'Yes' if dataset.enable_bucket else 'No'}"
        )

        if dataset.enable_bucket:
            details_lines.append(
                f"│ Bucket No Upscale   │ {'Yes' if dataset.bucket_no_upscale else 'No'}"
            )

        details_lines.append(
            f"│ Caption Extension   │ {dataset.caption_extension or 'None'}"
        )
        details_lines.append(
            f"│ Debug Mode          │ {'Yes' if dataset.debug_dataset else 'No'}"
        )

        # Type-specific parameters
        if is_image_dataset:
            details_lines.append(
                f"│ Image Directory     │ {dataset.image_directory or 'N/A'}"
            )
        else:
            details_lines.append(
                f"│ Video Directory     │ {dataset.video_directory or 'N/A'}"
            )
            details_lines.append(f"│ Target Frames       │ {dataset.target_frames}")
            details_lines.append(
                f"│ Frame Extraction    │ {dataset.frame_extraction or 'N/A'}"
            )
            details_lines.append(
                f"│ Frame Stride        │ {dataset.frame_stride or 'N/A'}"
            )
            details_lines.append(
                f"│ Frame Sample        │ {dataset.frame_sample or 'N/A'}"
            )
            details_lines.append(
                f"│ Max Frames          │ {dataset.max_frames or 'N/A'}"
            )
            details_lines.append(
                f"│ Source FPS          │ {dataset.source_fps or 'N/A'}"
            )

        details_lines.append(
            f"│ Cache Directory     │ {dataset.cache_directory or 'N/A'}"
        )
        details_lines.append("")  # Empty line between datasets

    return "\n".join(details_lines)


# if training is True, it will return a dataset group for training, otherwise for caching
def generate_dataset_group_by_blueprint(
    dataset_group_blueprint: DatasetGroupBlueprint,
    training: bool = False,
    load_pixels_for_batches: bool = False,
    prior_loss_weight: float = 1.0,
    num_timestep_buckets: Optional[int] = None,
) -> DatasetGroup:
    datasets: List[Union[ImageDataset, VideoDataset]] = []

    for i, dataset_blueprint in enumerate(dataset_group_blueprint.datasets):
        if dataset_blueprint.is_image_dataset:
            dataset_klass = ImageDataset
        else:
            dataset_klass = VideoDataset

        dataset = dataset_klass(**asdict(dataset_blueprint.params))
        datasets.append(dataset)

    # TODO: This assertion could be moved to a validation function for better separation of concerns
    cache_directories = [dataset.cache_directory for dataset in datasets]
    num_of_unique_cache_directories = len(set(cache_directories))
    if num_of_unique_cache_directories != len(cache_directories):
        raise ValueError(
            "cache directory should be unique for each dataset (note that cache directory is image/video directory if not specified)"
        )
    # REFACTOR: Extract cache directory validation to a separate method for better reusability

    # Display dataset information in a clean table format

    # Create and log the formatted table
    table_info = _format_dataset_info_table(datasets)
    logger.info("📊 Dataset Configuration Summary:")
    logger.info(f"\n{table_info}")

    # Log detailed configuration for all datasets
    detailed_info = _format_dataset_details_table(datasets)
    logger.info("📋 Detailed Dataset Configuration:")
    logger.info(f"\n{detailed_info}")

    # make buckets first because it determines the length of dataset
    # and set the same seed for all datasets
    seed = random.randint(
        0, 2**31
    )  # TODO: Magic number - consider making this configurable - REFACTOR: Extract to constant or config parameter
    for i, dataset in enumerate(datasets):
        # logger.info(f"[Dataset {i}]")  # TODO: Commented out logging - consider removing or enabling
        dataset.set_seed(seed)
        if training:
            if hasattr(dataset, "prepare_for_training"):
                import inspect

                sig = inspect.signature(dataset.prepare_for_training)
                kwargs = {}
                if "require_text_encoder_cache" in sig.parameters:
                    kwargs["require_text_encoder_cache"] = True
                if "load_pixels_for_control" in sig.parameters:
                    # This flag controls whether original pixels are included in batches
                    # for things like perceptual metrics or control preprocessing.
                    kwargs["load_pixels_for_control"] = load_pixels_for_batches
                if "prior_loss_weight" in sig.parameters:
                    kwargs["prior_loss_weight"] = prior_loss_weight
                if "num_timestep_buckets" in sig.parameters:
                    kwargs["num_timestep_buckets"] = num_timestep_buckets
                dataset.prepare_for_training(**kwargs)
            else:
                pass
        else:
            # Skip prepare_for_training during caching mode to avoid error messages
            # about missing cache files (since we're creating them)
            logger.info(
                f"📦 [Dataset {i}] Skipping training preparation during caching mode"
            )
            pass
        # REFACTOR: Consider extracting dataset preparation logic to a separate method for better organization

    return DatasetGroup(datasets)


def load_user_config(
    config_file: str,
) -> dict:  # Fixed: renamed parameter to avoid shadowing
    file_path: Path = Path(config_file)  # Fixed: use different variable name
    if not file_path.is_file():
        raise ValueError(f"file not found: {file_path}")

    if file_path.name.lower().endswith(".toml"):
        try:
            config = toml.load(file_path)
        except Exception:
            logger.error(
                f"Error on parsing TOML config file. Please check the format: {file_path}"
            )
            raise
    else:
        raise ValueError(f"not supported config file format: {file_path}")

    return config


def validate_dataset_config(
    config_file: str,
    argparse_namespace: Optional[argparse.Namespace] = None,
    test_dataset_creation: bool = True,
) -> bool:
    """
    Validates a dataset configuration file by loading, sanitizing, and optionally testing dataset creation.

    Args:
        config_file: Path to the configuration file
        argparse_namespace: Optional argparse namespace for additional parameters
        test_dataset_creation: Whether to test actual dataset group creation

    Returns:
        True if validation passes, raises exception if validation fails
    """
    logger.info(f"Validating dataset config: {config_file}")

    # Create default argparse namespace if not provided
    if argparse_namespace is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--debug_dataset", action="store_true")
        argparse_namespace = parser.parse_args([])

    logger.info("[argparse_namespace]")
    logger.info(f"{vars(argparse_namespace)}")

    # Load and validate user config
    user_config = load_user_config(config_file)

    logger.info("📝 [user_config]")
    logger.info(f"{user_config}")

    # Sanitize the config
    sanitizer = ConfigSanitizer()
    sanitized_user_config = sanitizer.sanitize_user_config(user_config)

    logger.info("🧹 [sanitized_user_config]")
    logger.info(f"{sanitized_user_config}")

    # Generate blueprint
    blueprint = BlueprintGenerator(sanitizer).generate(user_config, argparse_namespace)

    logger.info("📋 [blueprint]")
    logger.info(f"{blueprint}")

    # Optionally test dataset creation
    if test_dataset_creation:
        logger.info("Testing dataset group creation...")

        train_dataset_group = generate_dataset_group_by_blueprint(
            blueprint.train_dataset_group
        )

        if len(blueprint.val_dataset_group.datasets) > 0:
            val_dataset_group = generate_dataset_group_by_blueprint(
                blueprint.val_dataset_group
            )
            dataset_group = DatasetGroup(
                train_dataset_group.datasets + val_dataset_group.datasets
            )
        else:
            dataset_group = train_dataset_group

        logger.info(
            f"Successfully created dataset group with {len(dataset_group.datasets)} datasets"
        )

    logger.info("Dataset config validation completed successfully!")
    return True
