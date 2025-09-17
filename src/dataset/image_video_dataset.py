from concurrent.futures import ThreadPoolExecutor
import glob
import os
import json
import random
import time
from typing import Any, Optional, Sequence, Tuple, Union
import logging
import numpy as np
import torch
from PIL import Image

from dataset.buckets import BucketBatchManager, BucketSelector
from dataset.data_sources import (
    ContentDataSource,
    ImageDirectoryDatasource,
    VideoDirectoryDataSource,
)
from dataset.datasource_utils import resize_image_to_bucket
from dataset.frame_extraction import generate_crop_positions

from dataset.item_info import ItemInfo
from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)

TARGET_FPS_WAN = 16


def get_cache_postfix(target_model: Optional[str] = None) -> Tuple[str, str]:
    """
    Get latent and text encoder cache postfixes based on target model.

    Args:
        target_model: Model type ("wan21", "wan22", etc.)

    Returns:
        Tuple of (latent_postfix, text_encoder_postfix)
    """
    if target_model == "wan21":
        return "wan21", "wan21_te"
    elif target_model == "wan22":
        return "wan22", "wan22_te"
    else:
        # Default fallback (for backwards compatibility)
        return "wan2x", "wan2x_te"


# Keep these for backwards compatibility, but they will be overridden by dynamic values
LATENT_CACHE_POSTFIX = "wan2x"
TEXT_ENCODER_CACHE_POSTFIX = "wan2x_te"


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        resolution: Tuple[int, int] = (960, 544),
        caption_extension: Optional[str] = None,
        caption_dropout_rate: float = 0.0,
        batch_size: int = 1,
        num_repeats: int = 1,
        enable_bucket: bool = False,
        bucket_no_upscale: bool = False,
        cache_directory: Optional[str] = None,
        debug_dataset: bool = False,
        is_val: bool = False,
        target_model: Optional[str] = None,
        is_reg: bool = False,
    ):
        # TODO: REFACTOR - Hardcoded default resolution should be configurable
        self.resolution = resolution
        self.caption_extension = caption_extension
        self.caption_dropout_rate = (
            float(caption_dropout_rate) if caption_dropout_rate is not None else 0.0
        )
        self.batch_size = batch_size
        self.num_repeats = num_repeats
        self.enable_bucket = enable_bucket
        self.bucket_no_upscale = bucket_no_upscale
        self.cache_directory = cache_directory
        self.debug_dataset = debug_dataset
        self.is_val = is_val
        self.target_model = target_model
        self.is_reg = is_reg
        self.seed = None
        self.current_epoch = 0
        self.dataset_index: Optional[int] = None  # Set by DatasetGroup when applicable

        # Get dynamic cache postfixes based on target model
        self.latent_cache_postfix, self.text_encoder_cache_postfix = get_cache_postfix(
            target_model
        )

        if not self.enable_bucket:
            self.bucket_no_upscale = False

    def get_dataset_type(self) -> str:
        """Get the type of this dataset for logging purposes."""
        return self.__class__.__name__

    def get_dataset_identifier(self) -> str:
        """Get a human-readable identifier for this dataset including type and index."""
        dataset_type = self.get_dataset_type()

        # Add validation suffix if this is a validation dataset
        if self.is_val:
            dataset_type += "(val)"

        # Add index if set (used in DatasetGroup)
        if self.dataset_index is not None:
            return f"{dataset_type}[{self.dataset_index}]"
        else:
            return dataset_type

    def get_dataset_details(self) -> str:
        """Get detailed information about this dataset for logging."""
        details = []

        # Add directory information if available
        if hasattr(self, "image_directory") and getattr(self, "image_directory", None):
            details.append(f"dir={os.path.basename(getattr(self, 'image_directory'))}")
        elif hasattr(self, "video_directory") and getattr(
            self, "video_directory", None
        ):
            details.append(f"dir={os.path.basename(getattr(self, 'video_directory'))}")

        # Add item count if available
        if hasattr(self, "num_train_items") and getattr(self, "num_train_items", 0) > 0:
            details.append(f"items={getattr(self, 'num_train_items')}")

        # Add batch size
        details.append(f"batch_size={self.batch_size}")

        if details:
            return f" ({', '.join(details)})"
        return ""

    def set_dataset_index(self, index: int):
        """Set the dataset index (used by DatasetGroup)."""
        self.dataset_index = index

    def get_metadata(self) -> dict:
        metadata = {
            "resolution": self.resolution,
            "caption_extension": self.caption_extension,
            "batch_size_per_device": self.batch_size,
            "num_repeats": self.num_repeats,
            "enable_bucket": bool(self.enable_bucket),
            "bucket_no_upscale": bool(self.bucket_no_upscale),
        }
        return metadata

    def get_all_latent_cache_files(self):
        return glob.glob(
            os.path.join(
                self.cache_directory, f"*_{self.latent_cache_postfix}.safetensors"  # type: ignore
            )  # type: ignore
        )

    def get_all_text_encoder_output_cache_files(self):
        return glob.glob(
            os.path.join(
                self.cache_directory, f"*_{self.text_encoder_cache_postfix}.safetensors"  # type: ignore
            )  # type: ignore
        )

    def get_latent_cache_path(self, item_info: ItemInfo) -> str:
        """
        Returns the cache path for the latent tensor.

        item_info: ItemInfo object

        Returns:
            str: cache path

        cache_path is based on the item_key and the resolution.
        """
        w, h = item_info.original_size
        basename = os.path.splitext(os.path.basename(item_info.item_key))[0]
        assert self.cache_directory is not None, "cache_directory is required"
        return os.path.join(
            self.cache_directory,
            f"{basename}_{w:04d}x{h:04d}_{self.latent_cache_postfix}.safetensors",
        )

    def get_text_encoder_output_cache_path(self, item_info: ItemInfo) -> str:
        basename = os.path.splitext(os.path.basename(item_info.item_key))[0]
        assert self.cache_directory is not None, "cache_directory is required"
        return os.path.join(
            self.cache_directory,
            f"{basename}_{self.text_encoder_cache_postfix}.safetensors",
        )

    def retrieve_latent_cache_batches(self, num_workers: int):
        raise NotImplementedError

    def retrieve_text_encoder_output_cache_batches(self, num_workers: int):
        raise NotImplementedError

    def set_seed(self, seed: int):
        self.seed = seed

    def set_current_epoch(self, epoch, force_shuffle=None, reason=None):
        """
        Set the current epoch for the dataset with conservative shuffling logic.

        Conservative Shuffling Philosophy:
        - Shuffling should be the EXCEPTION, not the rule
        - Only shuffle for true sequential epoch progression in training datasets
        - Never shuffle validation datasets automatically
        - Be conservative about all sync operations, checkpoint resumes, etc.

        Args:
            epoch: Target epoch number
            force_shuffle: If True, force bucket shuffling regardless of logic
                         If False, explicitly disable shuffling
                         If None, use conservative logic (default)
            reason: Optional string describing why epoch is being set (for better logging)
        """
        if self.current_epoch == epoch:
            return  # No change needed

        # Get dataset identifier for enhanced logging
        dataset_id = self.get_dataset_identifier()
        dataset_details = self.get_dataset_details()

        # Build context for logging
        context = f"[{dataset_id}{dataset_details}] (current_epoch: {self.current_epoch}, epoch: {epoch})"
        if reason:
            context += f" - {reason}"

        # Handle explicit shuffle control
        if force_shuffle is True:
            logger.info(f"Force shuffling buckets {context}")
            self.current_epoch = epoch
            self.shuffle_buckets()
            return

        if force_shuffle is False:
            logger.debug(f"Shuffling explicitly disabled {context}")
            self.current_epoch = epoch
            return

        # Conservative shuffling logic - shuffling should be rare and intentional
        is_sequential_increment = epoch == self.current_epoch + 1
        is_initialization = self.current_epoch == 0 and epoch > 0

        # CRITICAL: Validation datasets should NEVER shuffle automatically
        if self.is_val:
            logger.debug(
                f"Validation dataset epoch update {context} - no shuffle (validation datasets never shuffle)"
            )
            self.current_epoch = epoch
            return

        # CONSERVATIVE: Only shuffle in very specific training scenarios

        # 1. Normal sequential training progression (most common case)
        if is_sequential_increment:
            # Only shuffle if this appears to be genuine training progression
            if reason and any(
                keyword in reason.lower()
                for keyword in [
                    "sync",
                    "validation",
                    "collator",
                    "worker",
                    "dataloader",
                    "checkpoint",
                    "resume",
                ]
            ):
                # This is likely a sync operation, not genuine training progression
                logger.debug(
                    f"Sequential increment but sync operation detected {context} - no shuffle"
                )
                self.current_epoch = epoch
            else:
                # Genuine training epoch progression
                logger.info(f"Training epoch progression {context} - shuffling")
                self.current_epoch = epoch
                self.shuffle_buckets()
            return

        # 2. Initialization (first epoch)
        if is_initialization:
            # Only shuffle if this appears to be genuine training start
            if reason and any(
                keyword in reason.lower()
                for keyword in [
                    "sync",
                    "validation",
                    "collator",
                    "worker",
                    "dataloader",
                ]
            ):
                logger.debug(
                    f"Initialization but sync operation detected {context} - no shuffle"
                )
                self.current_epoch = epoch
            else:
                logger.info(f"Training initialization {context} - shuffling")
                self.current_epoch = epoch
                self.shuffle_buckets()
            return

        # 3. ALL OTHER SCENARIOS: Default to NO shuffling (conservative approach)

        # Classify the scenario for appropriate logging
        if epoch > self.current_epoch:
            if epoch - self.current_epoch > 1:
                # Large forward jump
                if reason and any(
                    keyword in reason.lower()
                    for keyword in ["resume", "checkpoint", "restore", "load"]
                ):
                    logger.info(
                        f"Checkpoint resume detected {context} - no shuffle (preserving reproducibility)"
                    )
                elif reason and any(
                    keyword in reason.lower()
                    for keyword in [
                        "validation",
                        "sync",
                        "collator",
                        "worker",
                        "dataloader",
                    ]
                ):
                    logger.debug(f"Sync operation large jump {context} - no shuffle")
                else:
                    logger.warning(
                        f"Large epoch jump {context} - no shuffle (use force_shuffle=True if needed)"
                    )
            else:
                # Small forward jump (shouldn't happen with good logic above, but handle gracefully)
                logger.debug(f"Small forward jump {context} - no shuffle")
        elif epoch < self.current_epoch:
            # Backward jump
            logger.warning(
                f"Backward epoch change {context} - no shuffle (unusual scenario)"
            )
        else:
            # Same epoch (already handled above, but defensive programming)
            logger.debug(f"Same epoch update {context} - no shuffle")

        # In all cases above, just update epoch without shuffling
        self.current_epoch = epoch

    def set_current_step(self, step):
        dataset_id = self.get_dataset_identifier()
        logger.debug(f"[{dataset_id}] Setting current step: {step}")
        self.current_step = step

    def set_max_train_steps(self, max_train_steps):
        dataset_id = self.get_dataset_identifier()
        logger.debug(f"[{dataset_id}] Setting max train steps: {max_train_steps}")
        self.max_train_steps = max_train_steps

    def shuffle_buckets(self):
        raise NotImplementedError

    def __len__(self):
        return NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def _default_retrieve_text_encoder_output_cache_batches(
        self, datasource: ContentDataSource, batch_size: int, num_workers: int
    ):
        # TODO: REFACTOR - The ItemInfo creation with dummy values (0,0) is questionable
        datasource.set_caption_only(True)
        executor = ThreadPoolExecutor(max_workers=num_workers)

        data: list[ItemInfo] = []
        futures = []

        def aggregate_future(consume_all: bool = False):
            while len(futures) >= num_workers or (consume_all and len(futures) > 0):
                completed_futures = [future for future in futures if future.done()]
                if len(completed_futures) == 0:
                    if (
                        len(futures) >= num_workers or consume_all
                    ):  # to avoid adding too many futures
                        time.sleep(0.1)
                        continue
                    else:
                        break  # submit batch if possible

                for future in completed_futures:
                    item_key, caption = future.result()
                    item_info = ItemInfo(item_key, caption, (0, 0), (0, 0), is_reg=self.is_reg)  # type: ignore
                    item_info.text_encoder_output_cache_path = (
                        self.get_text_encoder_output_cache_path(item_info)
                    )
                    data.append(item_info)

                    futures.remove(future)

        def submit_batch(flush: bool = False):
            nonlocal data
            if len(data) >= batch_size or (len(data) > 0 and flush):
                batch = data[0:batch_size]
                if len(data) > batch_size:
                    data = data[batch_size:]
                else:
                    data = []
                return batch
            return None

        for fetch_op in datasource:
            future = executor.submit(fetch_op)
            futures.append(future)
            aggregate_future()
            while True:
                batch = submit_batch()
                if batch is None:
                    break
                yield batch

        aggregate_future(consume_all=True)
        while True:
            batch = submit_batch(flush=True)
            if batch is None:
                break
            yield batch

        executor.shutdown()


class ImageDataset(BaseDataset):
    def __init__(
        self,
        resolution: Tuple[int, int],
        caption_extension: Optional[str],
        batch_size: int,
        num_repeats: int,
        enable_bucket: bool,
        bucket_no_upscale: bool,
        image_directory: Optional[str] = None,
        cache_directory: Optional[str] = None,
        debug_dataset: bool = False,
        is_val: bool = False,
        load_control: bool = False,
        control_suffix: str = "_control",
        target_model: Optional[str] = None,
        mask_path: Optional[str] = None,
        is_reg: bool = False,
        caption_dropout_rate: float = 0.0,
    ):
        super().__init__(
            resolution=resolution,
            caption_extension=caption_extension,
            caption_dropout_rate=caption_dropout_rate,
            batch_size=batch_size,
            num_repeats=num_repeats,
            enable_bucket=enable_bucket,
            bucket_no_upscale=bucket_no_upscale,
            cache_directory=cache_directory,
            debug_dataset=debug_dataset,
            is_val=is_val,
            target_model=target_model,
            is_reg=is_reg,
        )

        self.image_directory = image_directory
        self.load_control = load_control
        self.control_suffix = control_suffix
        self.mask_path = mask_path

        if image_directory is not None:
            self.datasource = ImageDirectoryDatasource(
                image_directory,
                caption_extension,
            )
            # propagate caption dropout rate to datasource
            if hasattr(self, "caption_dropout_rate"):
                try:
                    self.datasource.set_caption_dropout_rate(self.caption_dropout_rate)
                except Exception:
                    pass

        else:
            raise ValueError("⛔ image_directory must be specified")

        if self.cache_directory is None:
            self.cache_directory = self.image_directory

        self.batch_manager = None
        self.num_train_items = 0

        # Set control settings if provided
        if hasattr(self, "load_control") and hasattr(self, "control_suffix"):
            self.datasource.set_control_settings(self.load_control, self.control_suffix)

        # Set mask settings if provided
        if hasattr(self.datasource, "set_mask_settings"):
            self.datasource.set_mask_settings(
                load_mask=bool(self.mask_path),
                mask_path=self.mask_path,
                default_mask_file=None,
            )

        # Update control availability after settings have been applied
        self.has_control = self.datasource.has_control

    def get_metadata(self):
        metadata = super().get_metadata()
        if self.image_directory is not None:
            metadata["image_directory"] = os.path.basename(self.image_directory)
        metadata["has_control"] = self.has_control
        return metadata

    def get_total_image_count(self):
        return len(self.datasource) if self.datasource.is_indexable() else None

    def retrieve_latent_cache_batches(self, num_workers: int):
        buckset_selector = BucketSelector(
            self.resolution,
            self.enable_bucket,
            self.bucket_no_upscale,
        )
        executor = ThreadPoolExecutor(max_workers=num_workers)

        batches: dict[tuple[int, int], list[ItemInfo]] = (
            {}
        )  # (width, height) -> [ItemInfo]
        futures = []

        # aggregate futures and sort by bucket resolution
        def aggregate_future(consume_all: bool = False):
            while len(futures) >= num_workers or (consume_all and len(futures) > 0):
                completed_futures = [future for future in futures if future.done()]
                if len(completed_futures) == 0:
                    if (
                        len(futures) >= num_workers or consume_all
                    ):  # to avoid adding too many futures
                        time.sleep(0.1)
                        continue
                    else:
                        break  # submit batch if possible

                for future in completed_futures:
                    original_size, item_key, image, caption, controls, masks = (
                        future.result()
                    )
                    bucket_height, bucket_width = image.shape[:2]
                    bucket_reso = (bucket_width, bucket_height)

                    # Create ItemInfo with control content if available
                    item_info = ItemInfo(
                        item_key=item_key,
                        caption=caption,
                        original_size=original_size,
                        bucket_size=bucket_reso,  # This should be a tuple # type: ignore
                        content=image,
                        latent_cache_path=self.get_latent_cache_path(
                            ItemInfo(item_key, caption, original_size, bucket_reso)  # type: ignore
                        ),
                        weight=1.0,  # Placeholder, will be updated later
                        control_content=controls,  # Add control content
                        mask_content=masks,  # Add mask content
                        is_reg=self.is_reg,
                    )
                    item_info.latent_cache_path = self.get_latent_cache_path(item_info)

                    # Add control content if available
                    if controls is not None and len(controls) > 0:
                        # For now, use the first control image
                        item_info.control_content = controls[0]

                    if bucket_reso not in batches:
                        batches[bucket_reso] = []
                    batches[bucket_reso].append(item_info)

                    futures.remove(future)

        # submit batch if some bucket has enough items
        def submit_batch(flush: bool = False):
            for key in batches:
                if len(batches[key]) >= self.batch_size or flush:
                    batch = batches[key][0 : self.batch_size]
                    if len(batches[key]) > self.batch_size:
                        batches[key] = batches[key][self.batch_size :]
                    else:
                        del batches[key]
                    return key, batch
            return None, None

        for fetch_op in self.datasource:
            # fetch and resize image in a separate thread
            def fetch_and_resize(
                op: callable,  # type: ignore
            ) -> tuple[
                tuple[int, int],
                str,
                Image.Image,
                str,
                Optional[np.ndarray],
                Optional[np.ndarray],
            ]:
                image_key, image, caption, controls, masks = op()
                image: Image.Image
                image_size = image.size

                bucket_reso = buckset_selector.get_bucket_resolution(image_size)
                image = resize_image_to_bucket(image, bucket_reso)  # type: ignore # returns np.ndarray
                resized_controls = None
                resized_masks = None
                if controls is not None:
                    # Resize control signal to match the bucket resolution
                    if isinstance(controls, Image.Image):
                        controls = controls.resize(
                            bucket_reso, Image.Resampling.LANCZOS
                        )
                        controls = np.array(controls)
                    elif isinstance(controls, np.ndarray):
                        # Assuming controls is already in the right format
                        # You might need to resize it here if needed
                        pass
                    resized_controls = controls

                if masks is not None:
                    # Resize mask to match the bucket resolution
                    if isinstance(masks, Image.Image):
                        masks = masks.resize(bucket_reso, Image.Resampling.LANCZOS)
                        masks = np.array(masks)
                    elif isinstance(masks, np.ndarray):
                        # Assuming masks is already in the right format
                        # You might need to resize it here if needed
                        pass
                    resized_masks = masks

                return (
                    image_size,
                    image_key,
                    image,
                    caption,
                    resized_controls,
                    resized_masks,
                )

            future = executor.submit(fetch_and_resize, fetch_op)
            futures.append(future)
            aggregate_future()
            while True:
                key, batch = submit_batch()
                if key is None:
                    break
                yield key, batch

        aggregate_future(consume_all=True)
        while True:
            key, batch = submit_batch(flush=True)
            if key is None:
                break
            yield key, batch

        executor.shutdown()

    def retrieve_text_encoder_output_cache_batches(self, num_workers: int):
        return self._default_retrieve_text_encoder_output_cache_batches(
            self.datasource, self.batch_size, num_workers
        )

    def prepare_for_training(
        self,
        load_pixels_for_control=False,
        require_text_encoder_cache=True,
        prior_loss_weight: float = 1.0,
        num_timestep_buckets: Optional[int] = None,
    ):
        dataset_id = self.get_dataset_identifier()
        logger.info(
            f"[{dataset_id}] Preparing for training (load_pixels_for_control={load_pixels_for_control}, require_text_encoder_cache={require_text_encoder_cache})"
        )

        bucket_selector = BucketSelector(
            self.resolution,
            self.enable_bucket,
            self.bucket_no_upscale,
        )

        # glob cache files
        latent_cache_files = glob.glob(
            os.path.join(
                self.cache_directory, f"*_{self.latent_cache_postfix}.safetensors"  # type: ignore
            )  # type: ignore
        )

        logger.info(
            f"[{dataset_id}] Found {len(latent_cache_files)} latent cache files"
        )

        # Enhanced error handling for missing cache files
        if len(latent_cache_files) == 0:
            logger.error(
                f"[{dataset_id}] ❌ No latent cache files found in {self.cache_directory}! "
                f"Expected files matching pattern: *_{self.latent_cache_postfix}.safetensors"
            )
            logger.error(
                f"[{dataset_id}] This will result in 0 training items and training failure."
            )
            logger.info("💡 To fix this issue:")
            logger.info("   1. Run latent caching first using the cache_latents script")
            logger.info("   2. Ensure the cache_directory path is correct")
            logger.info(
                "   3. Verify your dataset configuration matches the cached data"
            )

        # assign cache files to item info
        bucketed_item_info: dict[tuple[int, int], list[ItemInfo]] = {}
        skipped_items = 0
        processed_items = 0

        for cache_file in latent_cache_files:
            tokens = os.path.basename(cache_file).split("_")

            image_size = tokens[-2]  # 0000x0000
            image_width, image_height = map(int, image_size.split("x"))
            image_size = (image_width, image_height)

            item_key = "_".join(tokens[:-2])
            text_encoder_output_cache_file = os.path.join(
                self.cache_directory,  # type: ignore
                f"{item_key}_{self.text_encoder_cache_postfix}.safetensors",
            )

            # Check text encoder cache existence based on requirement
            text_encoder_cache_exists = os.path.exists(text_encoder_output_cache_file)
            if require_text_encoder_cache and not text_encoder_cache_exists:
                logger.warning(
                    f"[{dataset_id}] Text encoder cache missing for {item_key} - skipping item"
                )
                skipped_items += 1
                continue
            elif not require_text_encoder_cache and not text_encoder_cache_exists:
                # During latent caching phase, text encoder cache may not exist yet
                logger.debug(
                    f"Text encoder cache not yet available: {text_encoder_output_cache_file}"
                )

            bucket_reso = bucket_selector.get_bucket_resolution(image_size)

            # Load actual pixels if requested (for control LoRA)
            content = None
            if load_pixels_for_control:
                try:
                    # Reconstruct original image path from cache file name
                    image_path = self._get_original_image_path_from_cache(
                        cache_file, item_key
                    )
                    if image_path and os.path.exists(image_path):
                        # Load and resize image
                        with Image.open(image_path) as img:
                            image = img.convert("RGB")
                        content = resize_image_to_bucket(
                            image, bucket_reso
                        )  # returns np.ndarray
                        logger.debug(
                            f"Loaded pixels for control LoRA: {image_path}, shape: {content.shape}"
                        )
                    else:
                        logger.warning(
                            f"Could not find original image file for cache: {cache_file}"
                        )
                except Exception as e:
                    logger.warning(f"Failed to load pixels for {cache_file}: {e}")

            item_info = ItemInfo(
                item_key,
                "",
                image_size,
                bucket_reso,  # type: ignore
                content=content,
                latent_cache_path=cache_file,
                is_reg=self.is_reg,
            )
            # Only set text encoder cache path if it exists or is not required
            if text_encoder_cache_exists:
                item_info.text_encoder_output_cache_path = (
                    text_encoder_output_cache_file
                )
            else:
                item_info.text_encoder_output_cache_path = None

            bucket = bucketed_item_info.get(bucket_reso, [])
            for _ in range(self.num_repeats):
                bucket.append(item_info)
            bucketed_item_info[bucket_reso] = bucket
            processed_items += 1

        # Enhanced reporting
        if skipped_items > 0:
            logger.warning(
                f"[{dataset_id}] ⚠️  Skipped {skipped_items} items due to missing text encoder cache files"
            )
            if require_text_encoder_cache:
                logger.info("💡 To include these items:")
                logger.info("   1. Run text encoder caching for the missing files")
                logger.info(
                    "   2. Or set require_text_encoder_cache=False (not recommended for training)"
                )

        # prepare batch manager
        self.batch_manager = BucketBatchManager(
            bucketed_item_info, self.batch_size, prior_loss_weight
        )
        # Store per-epoch timestep bucketing preference on the batch manager if supported
        try:
            if hasattr(self.batch_manager, "set_num_timestep_buckets"):
                self.batch_manager.set_num_timestep_buckets(num_timestep_buckets)  # type: ignore
            elif hasattr(self.batch_manager, "num_timestep_buckets"):
                setattr(self.batch_manager, "num_timestep_buckets", num_timestep_buckets)  # type: ignore
        except Exception:
            pass

        self.batch_manager.show_bucket_info()

        self.num_train_items = sum(
            [len(bucket) for bucket in bucketed_item_info.values()]
        )

        # Final validation
        if self.num_train_items == 0:
            logger.error(
                f"[{dataset_id}] ❌ Dataset preparation resulted in 0 training items! "
                f"Training will fail. Please check cache files and dataset configuration."
            )
        else:
            logger.info(
                f"[{dataset_id}] ✅ Training preparation complete: {self.num_train_items} items across {len(bucketed_item_info)} buckets"
            )

    def shuffle_buckets(self):
        dataset_id = self.get_dataset_identifier()
        # set random seed for this epoch
        random.seed(self.seed + self.current_epoch)  # type: ignore
        logger.debug(
            f"[{dataset_id}] Shuffling buckets with seed={self.seed + self.current_epoch}"  # type: ignore
        )
        self.batch_manager.shuffle()  # type: ignore

    def __len__(self):
        if self.batch_manager is None:
            return 100  # dummy value
        return len(self.batch_manager)

    def __getitem__(self, idx):
        return self.batch_manager[idx]  # type: ignore

    def _get_original_image_path_from_cache(
        self, cache_file: str, item_key: str
    ) -> str:
        """Try to find the original image file that corresponds to a cache file."""
        # The item_key is derived from the original filename
        # Try common image extensions
        image_extensions = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"]

        # Check in the image directory
        if self.image_directory:
            for ext in image_extensions:
                potential_path = os.path.join(self.image_directory, item_key + ext)
                if os.path.exists(potential_path):
                    return potential_path

        # If not found in image_directory, try looking in subdirectories
        # This handles cases where images might be in nested folders
        cache_dir = os.path.dirname(cache_file)
        for root, dirs, files in os.walk(cache_dir):
            for ext in image_extensions:
                potential_file = item_key + ext
                if potential_file in files:
                    return os.path.join(root, potential_file)

        # Last resort: search more broadly
        if self.image_directory:
            for root, dirs, files in os.walk(self.image_directory):
                for ext in image_extensions:
                    potential_file = item_key + ext
                    if potential_file in files:
                        return os.path.join(root, potential_file)

        return ""

    def _process_mask_image(
        self, mask_image: Optional[Image.Image]
    ) -> Optional[torch.Tensor]:
        """Process mask image and convert to tensor for training.

        Args:
            mask_image: PIL Image in grayscale (L mode)

        Returns:
            Tensor with values between 0 and 1, where:
            - 1.0 (white) means train on this pixel
            - 0.0 (black) means mask it out
            - Values in between become weights between 0 and 1
        """
        if mask_image is None:
            return None

        # Convert PIL image to tensor
        mask_tensor = torch.from_numpy(np.array(mask_image)).float()

        # Normalize to [0, 1] range
        if mask_tensor.max() > 1.0:
            mask_tensor = mask_tensor / 255.0

        return mask_tensor

    def _process_mask_video(
        self, mask_frames: Optional[list[np.ndarray]]
    ) -> Optional[torch.Tensor]:
        """Process mask video frames and convert to tensor for training.

        Args:
            mask_frames: List of numpy arrays representing mask frames

        Returns:
            Tensor with values between 0 and 1, where:
            - 1.0 (white) means train on this pixel
            - 0.0 (black) means mask it out
            - Values in between become weights between 0 and 1
        """
        if mask_frames is None:
            return None

        # Convert list of numpy arrays to tensor
        mask_tensor = torch.from_numpy(np.stack(mask_frames)).float()

        # Normalize to [0, 1] range
        if mask_tensor.max() > 1.0:
            mask_tensor = mask_tensor / 255.0

        return mask_tensor


class VideoDataset(BaseDataset):
    def __init__(
        self,
        resolution: Tuple[int, int],
        caption_extension: Optional[str],
        batch_size: int,
        num_repeats: int,
        enable_bucket: bool,
        bucket_no_upscale: bool,
        frame_extraction: Optional[str] = "head",
        frame_stride: Optional[int] = 1,
        frame_sample: Optional[int] = 1,
        target_frames: Optional[list[int]] = None,
        max_frames: Optional[int] = None,
        source_fps: Optional[float] = None,
        video_directory: Optional[str] = None,
        cache_directory: Optional[str] = None,
        debug_dataset: bool = False,
        is_val: bool = False,
        load_control: bool = False,
        control_suffix: str = "_control",
        target_model: Optional[str] = None,
        mask_path: Optional[str] = None,
        is_reg: bool = False,
        caption_dropout_rate: float = 0.0,
    ):
        super().__init__(
            resolution=resolution,
            caption_extension=caption_extension,
            caption_dropout_rate=caption_dropout_rate,
            batch_size=batch_size,
            num_repeats=num_repeats,
            enable_bucket=enable_bucket,
            bucket_no_upscale=bucket_no_upscale,
            cache_directory=cache_directory,
            debug_dataset=debug_dataset,
            is_val=is_val,
            target_model=target_model,
            is_reg=is_reg,
        )

        self.video_directory = video_directory
        self.frame_extraction = frame_extraction
        self.frame_stride = frame_stride
        self.frame_sample = frame_sample
        self.target_frames = target_frames
        self.max_frames = max_frames
        self.source_fps = source_fps
        self.target_fps = TARGET_FPS_WAN
        self.load_control = load_control
        self.control_suffix = control_suffix
        self.mask_path = mask_path

        if video_directory is not None:
            self.datasource = VideoDirectoryDataSource(
                video_directory, caption_extension
            )
            # propagate caption dropout rate to datasource
            if hasattr(self, "caption_dropout_rate"):
                try:
                    self.datasource.set_caption_dropout_rate(self.caption_dropout_rate)  # type: ignore
                except Exception:
                    pass
            # Set up control loading if enabled
            if self.load_control:
                self.datasource.set_control_settings(
                    self.load_control, self.control_suffix
                )

            # Set up mask loading if enabled
            if hasattr(self.datasource, "set_mask_settings"):
                self.datasource.set_mask_settings(
                    load_mask=bool(self.mask_path),
                    mask_path=self.mask_path,
                    default_mask_file=None,
                )

        if self.frame_extraction == "uniform" and self.frame_sample == 1:
            self.frame_extraction = "head"
            logger.warning(
                "frame_sample is set to 1 for frame_extraction=uniform. frame_extraction is changed to head."
            )
        if self.frame_extraction == "head":
            # head extraction. we can limit the number of frames to be extracted
            self.datasource.set_start_and_end_frame(0, max(self.target_frames))  # type: ignore

        if self.cache_directory is None:
            self.cache_directory = self.video_directory

        self.batch_manager = None
        self.num_train_items = 0
        self.has_control = self.datasource.has_control

    def get_metadata(self):
        metadata = super().get_metadata()
        if self.video_directory is not None:
            metadata["video_directory"] = os.path.basename(self.video_directory)

        metadata["frame_extraction"] = self.frame_extraction
        metadata["frame_stride"] = self.frame_stride
        metadata["frame_sample"] = self.frame_sample
        metadata["target_frames"] = self.target_frames
        metadata["max_frames"] = self.max_frames
        metadata["source_fps"] = self.source_fps

        return metadata

    def retrieve_latent_cache_batches(self, num_workers: int):
        # Keep mask loading enabled so that we can pre-cache masks to `_mask.safetensors` alongside latents.
        # This path will forward per-sample masks through `ItemInfo(mask_content=...)` so the collator can save them.
        buckset_selector = BucketSelector(self.resolution)
        self.datasource.set_bucket_selector(buckset_selector)
        if self.source_fps is not None:
            self.datasource.set_source_and_target_fps(self.source_fps, self.target_fps)
        else:
            self.datasource.set_source_and_target_fps(None, None)  # no conversion

        executor = ThreadPoolExecutor(max_workers=num_workers)

        # key: (width, height, frame_count) and optional latent_window_size, value: [ItemInfo]
        batches: dict[tuple[Any], list[ItemInfo]] = {}
        futures = []

        def aggregate_future(consume_all: bool = False):
            while len(futures) >= num_workers or (consume_all and len(futures) > 0):
                completed_futures = [future for future in futures if future.done()]
                if len(completed_futures) == 0:
                    if (
                        len(futures) >= num_workers or consume_all
                    ):  # to avoid adding too many futures
                        time.sleep(0.1)
                        continue
                    else:
                        break  # submit batch if possible

                for future in completed_futures:
                    original_frame_size, video_key, video, caption, control, mask = (
                        future.result()
                    )

                    frame_count = len(video)
                    video = np.stack(video, axis=0)
                    height, width = video.shape[1:3]
                    bucket_reso = (width, height)  # already resized

                    # process control/mask videos if available
                    control_video = control
                    mask_video = None
                    if mask is not None:
                        try:
                            mask = [
                                resize_image_to_bucket(frame, bucket_reso)
                                for frame in mask
                            ]
                            # Ensure grayscale (H,W). If frames are RGB, take first channel
                            mask = [f[:, :, 0] if f.ndim == 3 else f for f in mask]
                            mask_video = mask
                        except Exception as e:
                            logger.warning(
                                f"Failed to resize mask video for {video_key}: {e}"
                            )

                    # Frame window generation extracted to helper for readability and extensibility
                    crop_pos_and_frames = generate_crop_positions(
                        frame_count=frame_count,
                        target_frames=self.target_frames,
                        mode=self.frame_extraction,  # type: ignore
                        frame_stride=self.frame_stride,
                        frame_sample=self.frame_sample,
                        max_frames=self.max_frames,
                    )

                    for crop_pos, target_frame in crop_pos_and_frames:
                        cropped_video = video[crop_pos : crop_pos + target_frame]
                        body, ext = os.path.splitext(video_key)
                        item_key = f"{body}_{crop_pos:05d}-{target_frame:03d}{ext}"
                        batch_key = (
                            *bucket_reso,
                            target_frame,
                        )  # bucket_reso with frame_count

                        # crop control video if available
                        cropped_control = None
                        if control_video is not None:
                            cropped_control = control_video[
                                crop_pos : crop_pos + target_frame
                            ]
                            try:
                                cropped_control = np.stack(cropped_control, axis=0)
                            except Exception:
                                pass

                        # crop mask video if available
                        cropped_mask = None
                        if mask_video is not None:
                            try:
                                cropped_mask = mask_video[
                                    crop_pos : crop_pos + target_frame
                                ]
                                cropped_mask = np.stack(cropped_mask, axis=0)
                            except Exception:
                                cropped_mask = None

                        item_info = ItemInfo(
                            item_key,
                            caption,
                            original_frame_size,
                            batch_key,  # type: ignore
                            frame_count=target_frame,
                            content=cropped_video,
                            control_content=cropped_control,
                            mask_content=cropped_mask,
                            is_reg=self.is_reg,
                        )
                        item_info.latent_cache_path = self.get_latent_cache_path(
                            item_info
                        )

                        batch = batches.get(batch_key, [])  # type: ignore
                        batch.append(item_info)
                        batches[batch_key] = batch  # type: ignore

                    futures.remove(future)

        def submit_batch(flush: bool = False):
            for key in batches:
                if len(batches[key]) >= self.batch_size or flush:
                    batch = batches[key][0 : self.batch_size]
                    if len(batches[key]) > self.batch_size:
                        batches[key] = batches[key][self.batch_size :]
                    else:
                        del batches[key]
                    return key, batch
            return None, None

        for operator in self.datasource:

            def fetch_and_resize(
                op: callable,  # type: ignore
            ) -> tuple[
                tuple[int, int],
                str,
                list[np.ndarray],
                str,
                Optional[list[np.ndarray]],
                Optional[list[np.ndarray]],
            ]:
                try:
                    result = op()

                    if (
                        len(result) == 3
                    ):  # for backward compatibility TODO remove this in the future
                        video_key, video, caption = result
                        control = None
                    elif len(result) == 4:
                        video_key, video, caption, control = result
                    elif len(result) == 5:
                        video_key, video, caption, control, mask = result
                    else:
                        raise ValueError(
                            f"Unexpected number of values from datasource: {len(result)}"
                        )

                    # Validate that video is a list of numpy arrays
                    if not isinstance(video, list) or len(video) == 0:
                        raise ValueError(
                            f"Invalid video data for {video_key}: expected list of numpy arrays, got {type(video)}"
                        )

                    if not isinstance(video[0], np.ndarray):
                        raise ValueError(
                            f"Invalid video frame data for {video_key}: expected numpy array, got {type(video[0])}"
                        )

                    video: list[np.ndarray]
                    frame_size = (video[0].shape[1], video[0].shape[0])

                    # resize if necessary
                    bucket_reso = buckset_selector.get_bucket_resolution(frame_size)
                    video = [
                        resize_image_to_bucket(frame, bucket_reso) for frame in video
                    ]

                    # resize control if necessary
                    if control is not None:
                        control = [
                            resize_image_to_bucket(frame, bucket_reso)
                            for frame in control
                        ]

                    # resize mask if necessary
                    if "mask" in locals() and mask is not None:
                        mask = [
                            resize_image_to_bucket(frame, bucket_reso) for frame in mask
                        ]
                        # Ensure grayscale (H, W)
                        mask = [
                            frame[:, :, 0] if frame.ndim == 3 else frame
                            for frame in mask
                        ]

                    return (
                        frame_size,
                        video_key,
                        video,
                        caption,
                        control,
                        mask if "mask" in locals() else None,
                    )
                except Exception as e:
                    logger.error(f"Error processing video: {e}")
                    raise

            future = executor.submit(fetch_and_resize, operator)
            futures.append(future)
            aggregate_future()
            while True:
                key, batch = submit_batch()
                if key is None:
                    break
                yield key, batch

        aggregate_future(consume_all=True)
        while True:
            key, batch = submit_batch(flush=True)
            if key is None:
                break
            yield key, batch

        executor.shutdown()

    def retrieve_text_encoder_output_cache_batches(self, num_workers: int):
        return self._default_retrieve_text_encoder_output_cache_batches(
            self.datasource, self.batch_size, num_workers
        )

    def prepare_for_training(
        self,
        load_pixels_for_control=False,
        require_text_encoder_cache=True,
        prior_loss_weight: float = 1.0,
    ):
        dataset_id = self.get_dataset_identifier()
        logger.info(
            f"[{dataset_id}] Preparing for training (load_pixels_for_control={load_pixels_for_control}, require_text_encoder_cache={require_text_encoder_cache})"
        )

        bucket_selector = BucketSelector(
            self.resolution,
            self.enable_bucket,
            self.bucket_no_upscale,
        )

        # glob cache files
        latent_cache_files = glob.glob(
            os.path.join(
                self.cache_directory, f"*_{self.latent_cache_postfix}.safetensors"  # type: ignore
            )  # type: ignore
        )

        logger.info(
            f"[{dataset_id}] Found {len(latent_cache_files)} latent cache files"
        )

        # assign cache files to item info
        bucketed_item_info: dict[tuple[int, int, int], list[ItemInfo]] = (
            {}
        )  # (width, height, frame_count) -> [ItemInfo]
        for cache_file in latent_cache_files:
            tokens = os.path.basename(cache_file).split("_")

            image_size = tokens[-2]  # 0000x0000
            image_width, image_height = map(int, image_size.split("x"))
            image_size = (image_width, image_height)

            frame_pos, frame_count = tokens[-3].split("-")[
                :2
            ]  # "00000-000", or optional section index "00000-000-00"
            frame_pos, frame_count = int(frame_pos), int(frame_count)

            item_key = "_".join(tokens[:-3])
            text_encoder_output_cache_file = os.path.join(
                self.cache_directory,  # type: ignore
                f"{item_key}_{self.text_encoder_cache_postfix}.safetensors",
            )

            # Check text encoder cache existence based on requirement
            text_encoder_cache_exists = os.path.exists(text_encoder_output_cache_file)
            if require_text_encoder_cache and not text_encoder_cache_exists:
                logger.warning(
                    f"Text encoder output cache file not found: {text_encoder_output_cache_file}"
                )
                continue
            elif not require_text_encoder_cache and not text_encoder_cache_exists:
                # During latent caching phase, text encoder cache may not exist yet
                logger.debug(
                    f"Text encoder cache not yet available: {text_encoder_output_cache_file}"
                )

            bucket_reso = bucket_selector.get_bucket_resolution(image_size)
            bucket_reso = (*bucket_reso, frame_count)

            # Load actual video pixels if requested (for control LoRA)
            content = None
            if load_pixels_for_control:
                logger.debug(f"Loading video pixels for control LoRA: {cache_file}")
                try:
                    # Reconstruct original video path from cache file name
                    video_path = self._get_original_video_path_from_cache(
                        cache_file, item_key
                    )
                    logger.debug(f"Reconstructed video path: {video_path}")
                    logger.debug(
                        f"Video path exists: {os.path.exists(video_path) if video_path else False}"
                    )
                    if video_path and os.path.exists(video_path):
                        import decord

                        # Load and resize video frames
                        vr = decord.VideoReader(video_path)
                        # Extract the same frames that were cached (frame_pos to frame_pos+frame_count)
                        frames = vr[frame_pos : frame_pos + frame_count]

                        # Convert entire NDArray to numpy first, then resize each frame
                        frames_np = frames.asnumpy()  # Convert entire NDArray to numpy
                        video_frames = []
                        for i in range(frames_np.shape[0]):
                            frame_np = frames_np[i]  # Now we can index the numpy array
                            resized_frame = resize_image_to_bucket(
                                frame_np, bucket_reso[:2]
                            )
                            video_frames.append(resized_frame)

                        # Stack frames into single array: (F, H, W, C)
                        content = np.stack(video_frames, axis=0)
                        logger.debug(
                            f"Loaded video pixels for control LoRA: {video_path}, shape: {content.shape}"
                        )
                    else:
                        logger.warning(
                            f"Could not find original video file for cache: {cache_file}"
                        )
                except Exception as e:
                    logger.warning(f"Failed to load video pixels for {cache_file}: {e}")

            item_info = ItemInfo(
                item_key,
                "",
                image_size,
                bucket_reso,  # type: ignore
                frame_count=frame_count,
                content=content,
                latent_cache_path=cache_file,
                control_content=(
                    content if load_pixels_for_control and content is not None else None
                ),
                is_reg=self.is_reg,
            )
            # Only set text encoder cache path if it exists or is not required
            if text_encoder_cache_exists:
                item_info.text_encoder_output_cache_path = (
                    text_encoder_output_cache_file
                )
            else:
                item_info.text_encoder_output_cache_path = None

            bucket = bucketed_item_info.get(bucket_reso, [])
            for _ in range(self.num_repeats):
                bucket.append(item_info)
            bucketed_item_info[bucket_reso] = bucket

        # prepare batch manager
        self.batch_manager = BucketBatchManager(bucketed_item_info, self.batch_size, prior_loss_weight)  # type: ignore

        self.batch_manager.show_bucket_info()

        self.num_train_items = sum(
            [len(bucket) for bucket in bucketed_item_info.values()]
        )

        logger.info(
            f"[{dataset_id}] Training preparation complete: {self.num_train_items} items across {len(bucketed_item_info)} buckets"
        )

    def shuffle_buckets(self):
        dataset_id = self.get_dataset_identifier()
        # set random seed for this epoch
        random.seed(self.seed + self.current_epoch)  # type: ignore
        logger.debug(
            f"[{dataset_id}] Shuffling buckets with seed={self.seed + self.current_epoch}"  # type: ignore
        )
        self.batch_manager.shuffle()  # type: ignore

    def __len__(self):
        if self.batch_manager is None:
            return 100  # dummy value
        return len(self.batch_manager)

    def __getitem__(self, idx):
        return self.batch_manager[idx]  # type: ignore

    def _get_original_video_path_from_cache(
        self, cache_file: str, item_key: str
    ) -> str:
        """Try to find the original video file that corresponds to a cache file."""
        # Try common video extensions
        video_extensions = [".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"]

        # Check in the video directory
        if self.video_directory:
            for ext in video_extensions:
                potential_path = os.path.join(self.video_directory, item_key + ext)
                if os.path.exists(potential_path):
                    return potential_path

        # If not found in video_directory, try looking in subdirectories
        cache_dir = os.path.dirname(cache_file)
        for root, dirs, files in os.walk(cache_dir):
            for ext in video_extensions:
                potential_file = item_key + ext
                if potential_file in files:
                    return os.path.join(root, potential_file)

        # Last resort: search more broadly
        if self.video_directory:
            for root, dirs, files in os.walk(self.video_directory):
                for ext in video_extensions:
                    potential_file = item_key + ext
                    if potential_file in files:
                        return os.path.join(root, potential_file)

        return ""

    def validate_resume_compatibility(
        self, expected_num_items: Optional[int] = None
    ) -> bool:
        """
        Validate if this dataset is compatible with a resumed training session.

        Args:
            expected_num_items: Expected number of training items from previous session

        Returns:
            True if compatible, False if major incompatibilities detected
        """
        dataset_id = self.get_dataset_identifier()

        if expected_num_items is None:
            logger.debug(
                f"[{dataset_id}] No expected item count provided - skipping compatibility check"
            )
            return True

        # Check if dataset has been prepared
        if not hasattr(self, "num_train_items") or self.num_train_items == 0:
            logger.error(
                f"[{dataset_id}] Dataset not prepared or has 0 items! "
                f"This will cause training failure. Please ensure cache files exist for this dataset."
            )
            return False

        # Check for significant size differences
        size_ratio = (
            self.num_train_items / expected_num_items if expected_num_items > 0 else 0
        )

        if size_ratio < 0.1 or size_ratio > 10.0:
            logger.error(
                f"[{dataset_id}] Major dataset size mismatch detected! "
                f"Expected ~{expected_num_items} items, but found {self.num_train_items} items "
                f"(ratio: {size_ratio:.2f}). This will break epoch/step calculations during resume."
            )
            return False
        elif size_ratio < 0.5 or size_ratio > 2.0:
            logger.warning(
                f"[{dataset_id}] Significant dataset size change detected! "
                f"Expected ~{expected_num_items} items, but found {self.num_train_items} items "
                f"(ratio: {size_ratio:.2f}). Training will continue but epoch/step calculations may be affected."
            )
        else:
            logger.info(
                f"[{dataset_id}] Dataset size compatible: {self.num_train_items} items "
                f"(expected ~{expected_num_items}, ratio: {size_ratio:.2f})"
            )

        return True

    def get_cache_compatibility_info(self) -> dict:
        """Get information about cache file compatibility for debugging."""
        dataset_id = self.get_dataset_identifier()
        info = {
            "dataset_id": dataset_id,
            "cache_directory": self.cache_directory,
            "latent_cache_files": 0,
            "text_encoder_cache_files": 0,
            "num_train_items": getattr(self, "num_train_items", 0),
            "has_batch_manager": self.batch_manager is not None,
        }

        if self.cache_directory and os.path.exists(self.cache_directory):
            # Count cache files
            latent_pattern = os.path.join(
                self.cache_directory, f"*_{self.latent_cache_postfix}.safetensors"
            )
            text_pattern = os.path.join(
                self.cache_directory, f"*_{self.text_encoder_cache_postfix}.safetensors"
            )

            info["latent_cache_files"] = len(glob.glob(latent_pattern))
            info["text_encoder_cache_files"] = len(glob.glob(text_pattern))

        return info


class DatasetGroup(torch.utils.data.ConcatDataset):
    # TODO: REFACTOR - Type annotation conflict with parent class
    # The parent class already has a 'datasets' attribute with different type
    def __init__(self, datasets: Sequence[Union[ImageDataset, VideoDataset]]):
        super().__init__(datasets)
        self.datasets: list[Union[ImageDataset, VideoDataset]] = datasets  # type: ignore
        self.num_train_items = 0

        # Set indices for each dataset to enable better logging
        for i, dataset in enumerate(self.datasets):
            dataset.set_dataset_index(i)
            self.num_train_items += dataset.num_train_items

        # Log initial group composition
        logger.info(f"📦 [DatasetGroup] Created with {len(self.datasets)} datasets:")
        for i, dataset in enumerate(self.datasets):
            dataset_id = dataset.get_dataset_identifier()
            dataset_details = dataset.get_dataset_details()
            logger.info(f"   {dataset_id}{dataset_details}")

    def get_dataset_identifier(self) -> str:
        """Get identifier for the dataset group."""
        return f"DatasetGroup({len(self.datasets)} datasets)"

    def set_current_epoch(self, epoch, force_shuffle=None, reason=None):
        """Set current epoch for all datasets in the group."""
        group_id = self.get_dataset_identifier()
        logger.debug(
            f"[{group_id}] Setting epoch {epoch} for all datasets"
            + (f" - {reason}" if reason else "")
        )

        for i, dataset in enumerate(self.datasets):
            dataset.set_current_epoch(epoch, force_shuffle=force_shuffle, reason=reason)

    def set_current_step(self, step):
        group_id = self.get_dataset_identifier()
        logger.debug(f"[{group_id}] Setting step {step} for all datasets")

        for i, dataset in enumerate(self.datasets):
            dataset.set_current_step(step)

    def set_max_train_steps(self, max_train_steps):
        group_id = self.get_dataset_identifier()
        logger.debug(
            f"[{group_id}] Setting max train steps {max_train_steps} for all datasets"
        )

        for i, dataset in enumerate(self.datasets):
            dataset.set_max_train_steps(max_train_steps)

    def validate_resume_compatibility(
        self, expected_dataset_info: Optional[dict] = None
    ) -> bool:
        """
        Validate if this dataset group is compatible with a resumed training session.

        Args:
            expected_dataset_info: Dictionary with expected dataset information from previous session

        Returns:
            True if compatible, False if major incompatibilities detected
        """
        group_id = self.get_dataset_identifier()

        if expected_dataset_info is None:
            logger.debug(
                f"[{group_id}] No expected dataset info provided - skipping compatibility check"
            )
            return True

        expected_total_items = expected_dataset_info.get("total_items", 0)
        expected_dataset_count = expected_dataset_info.get("dataset_count", 0)

        # Check dataset count mismatch
        if len(self.datasets) != expected_dataset_count:
            logger.error(
                f"[{group_id}] Dataset count mismatch! "
                f"Expected {expected_dataset_count} datasets, but found {len(self.datasets)} datasets. "
                f"This indicates a different dataset configuration."
            )
            return False

        # Validate individual datasets
        all_compatible = True
        expected_per_dataset = expected_dataset_info.get("per_dataset_items", [])

        for i, dataset in enumerate(self.datasets):
            expected_items = (
                expected_per_dataset[i] if i < len(expected_per_dataset) else None
            )
            if not dataset.validate_resume_compatibility(expected_items):  # type: ignore
                all_compatible = False

        # Check total items
        if expected_total_items > 0:
            total_ratio = self.num_train_items / expected_total_items
            if total_ratio < 0.1 or total_ratio > 10.0:
                logger.error(
                    f"[{group_id}] Major total item count mismatch! "
                    f"Expected ~{expected_total_items} total items, but found {self.num_train_items} items "
                    f"(ratio: {total_ratio:.2f})"
                )
                all_compatible = False

        if all_compatible:
            logger.info(f"[{group_id}] Resume compatibility validation passed ✅")
        else:
            logger.error(f"[{group_id}] Resume compatibility validation failed ❌")

        return all_compatible

    def get_dataset_info_for_resume(self) -> dict:
        """Get dataset information that can be saved for resume validation."""
        return {
            "total_items": self.num_train_items,
            "dataset_count": len(self.datasets),
            "per_dataset_items": [
                getattr(ds, "num_train_items", 0) for ds in self.datasets
            ],
            "dataset_types": [ds.get_dataset_type() for ds in self.datasets],
            "cache_info": [ds.get_cache_compatibility_info() for ds in self.datasets],  # type: ignore
        }

    def log_dataset_compatibility_info(self):
        """Log detailed compatibility information for debugging."""
        group_id = self.get_dataset_identifier()
        logger.info(f"[{group_id}] 📊 Dataset Compatibility Information:")

        for i, dataset in enumerate(self.datasets):
            cache_info = dataset.get_cache_compatibility_info()  # type: ignore
            dataset_id = cache_info["dataset_id"]

            logger.info(f"   Dataset {i}: {dataset_id}")
            logger.info(f"     Items: {cache_info['num_train_items']}")
            logger.info(f"     Latent caches: {cache_info['latent_cache_files']}")
            logger.info(
                f"     Text encoder caches: {cache_info['text_encoder_cache_files']}"
            )
            logger.info(f"     Cache directory: {cache_info['cache_directory']}")
            logger.info(f"     Ready for training: {cache_info['has_batch_manager']}")

        logger.info(f"   Total items across all datasets: {self.num_train_items}")


def save_dataset_metadata_for_resume(
    dataset_group: "DatasetGroup", output_dir: str, step: int
):
    """
    Save dataset metadata for resume validation.
    This should be called during checkpoint saving.
    """
    try:

        metadata = {
            "step": step,
            "dataset_info": dataset_group.get_dataset_info_for_resume(),
            "timestamp": time.time(),
        }

        metadata_path = os.path.join(output_dir, "dataset_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.debug(
            f"💾 Saved dataset metadata for resume validation: {metadata_path}"
        )

    except Exception as e:
        logger.warning(f"⚠️  Failed to save dataset metadata for resume validation: {e}")


def load_dataset_metadata_for_resume(resume_dir: str) -> Optional[dict]:
    """
    Load dataset metadata for resume validation.
    This should be called during checkpoint resuming.
    """
    try:
        import json

        metadata_path = os.path.join(resume_dir, "dataset_metadata.json")
        if not os.path.exists(metadata_path):
            logger.debug(
                f"No dataset metadata found for resume validation: {metadata_path}"
            )
            return None

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        logger.info(
            f"📂 Loaded dataset metadata for resume validation from: {metadata_path}"
        )
        return metadata.get("dataset_info")

    except Exception as e:
        logger.warning(f"⚠️  Failed to load dataset metadata for resume validation: {e}")
        return None


def validate_dataset_for_resume(
    dataset_group: "DatasetGroup",
    resume_dir: str,
    allow_dataset_change: bool = False,
    reset_training_state: bool = False,
    auto_handle_changes: bool = True,
) -> tuple[bool, bool, Optional[dict]]:
    """
    Intelligent dataset compatibility validation for resume.

    Args:
        dataset_group: The dataset group to validate
        resume_dir: Directory containing the resume state
        allow_dataset_change: If True, allow intentional dataset changes with warnings (legacy)
        reset_training_state: If True, suggest resetting epochs/steps for new dataset (legacy)
        auto_handle_changes: If True, automatically handle dataset changes intelligently (default)

    Returns:
        Tuple of (is_valid_to_continue, dataset_changed, expected_info)
        - is_valid_to_continue: Whether training can continue safely
        - dataset_changed: Whether the dataset is different from before
        - expected_info: Previous dataset information (if available)
    """
    expected_info = load_dataset_metadata_for_resume(resume_dir)
    if expected_info is None:
        logger.info(
            "🔍 No previous dataset metadata found - assuming fresh training start"
        )
        return True, False, None

    logger.info("🔍 Validating dataset compatibility for resume...")
    dataset_group.log_dataset_compatibility_info()

    is_compatible = dataset_group.validate_resume_compatibility(expected_info)

    if is_compatible:
        # Datasets are compatible - continue normally
        logger.info(
            "✅ Dataset compatibility validation passed - resuming training normally"
        )
        logger.info("📊 Same dataset detected - preserving training progress")
        return True, False, expected_info

    # Datasets are different - handle automatically if enabled
    if auto_handle_changes:
        logger.info("🤖 AUTOMATIC DATASET CHANGE DETECTION:")
        logger.info(
            f"   Previous dataset: {expected_info.get('total_items', 'unknown')} total items, {expected_info.get('dataset_count', 'unknown')} datasets"
        )
        logger.info(
            f"   Current dataset: {dataset_group.num_train_items} total items, {len(dataset_group.datasets)} datasets"
        )

        logger.info(
            "🔄 DATASET CHANGE STRATEGY: Always reset training state for safety"
        )
        logger.info("   • Different dataset detected → Reset training progress")
        logger.info("   • This ensures proper epoch/step calculations")
        logger.info("   • LoRA weights and optimizer state will be preserved")

        logger.info("🎯 LoRA Resume with Different Dataset - What's preserved:")
        logger.info("   ✅ LoRA network weights (your trained adaptations)")
        logger.info("   ✅ Optimizer state (momentum, learning rate history)")
        logger.info("   ✅ Training checkpoints and model state")
        logger.info("   🔄 Training progress will be reset (epoch/step counters)")
        logger.info("   🔄 This ensures proper training behavior with the new dataset")

        logger.info("")
        logger.info("💡 This conservative approach ensures:")
        logger.info("   • No training errors or crashes")
        logger.info("   • Proper learning rate scheduling")
        logger.info("   • Correct epoch/batch calculations")
        logger.info("   • Clean start with new data while keeping LoRA adaptations")

        return True, True, expected_info

    # Auto-handling disabled, use legacy behavior
    if allow_dataset_change:
        logger.warning("⚠️  Dataset change detected - using legacy manual handling")
        return True, True, expected_info
    else:
        logger.error("❌ Dataset compatibility validation failed!")
        logger.info(
            "💡 The dataset has changed significantly from the previous training session."
        )
        logger.info(
            "💡 This is automatically handled in most cases, but validation failed."
        )
        logger.info("💡 Please check your dataset configuration or cache files.")
        return False, True, expected_info


def handle_dataset_change_for_resume_auto(
    expected_info: dict, actual_dataset_group: "DatasetGroup", global_step: int
) -> tuple[int, int, bool]:
    """
    Automatically handle training state adjustments when resuming with a different dataset.
    Always resets training state for safety when datasets differ.

    Args:
        expected_info: Previous dataset information
        actual_dataset_group: Current dataset group
        global_step: Current global step from checkpoint

    Returns:
        Tuple of (adjusted_global_step, suggested_epoch_start, reset_applied)
    """
    prev_total_items = expected_info.get("total_items", 0)
    curr_total_items = actual_dataset_group.num_train_items

    logger.info("🔧 Handling dataset change for resume:")
    logger.info(f"   Previous dataset size: {prev_total_items} items")
    logger.info(f"   Current dataset size: {curr_total_items} items")
    logger.info(f"   Checkpoint global step: {global_step}")

    # Always reset training state when datasets are different
    logger.info("🔄 RESETTING TRAINING STATE:")
    logger.info("   Strategy: Conservative reset for all dataset changes")

    adjusted_step = 0
    suggested_epoch = 0
    reset_applied = True

    logger.info("   - Global step reset to 0")
    logger.info("   - Epoch reset to 0")
    logger.info("   - LoRA weights preserved from checkpoint")
    logger.info("   - Optimizer state preserved from checkpoint")

    # Calculate size change info for user awareness
    if prev_total_items > 0 and curr_total_items > 0:
        size_ratio = curr_total_items / prev_total_items
        if size_ratio < 0.8:
            logger.info(f"   📉 New dataset is smaller ({size_ratio:.2f}x)")
        elif size_ratio > 1.25:
            logger.info(f"   📈 New dataset is larger ({size_ratio:.2f}x)")
        else:
            logger.info(f"   📊 Similar dataset size ({size_ratio:.2f}x)")

    logger.info("✅ Training state reset complete")
    logger.info("💡 Training will start fresh with your preserved LoRA adaptations")
    logger.info("💡 This ensures clean epoch/step calculations for the new dataset")

    return adjusted_step, suggested_epoch, reset_applied


def generate_safe_output_name(
    original_output_name: str,
    expected_info: dict,
    current_dataset_group: "DatasetGroup",
    strategy: str = "auto",
) -> str:
    """
    Generate a safe output name when dataset changes to prevent overwriting existing checkpoints.

    Args:
        original_output_name: The original output name from config
        expected_info: Previous dataset information
        current_dataset_group: Current dataset group
        strategy: Naming strategy ("auto", "timestamp", "sequential", "descriptive")

    Returns:
        Safe output name that won't conflict with existing checkpoints
    """
    import time
    from datetime import datetime

    if strategy == "timestamp":
        # Add timestamp suffix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = f"{original_output_name}_dataset_change_{timestamp}"

    elif strategy == "sequential":
        # Add sequential version number
        base_name = original_output_name
        version = 2
        safe_name = f"{base_name}_v{version}"

        # Check if this name already exists and increment if needed
        # (This would need file system checking in real implementation)

    elif strategy == "descriptive":
        # Add descriptive suffix based on dataset characteristics
        prev_items = expected_info.get("total_items", 0)
        curr_items = current_dataset_group.num_train_items
        prev_datasets = expected_info.get("dataset_count", 0)
        curr_datasets = len(current_dataset_group.datasets)

        if curr_items > prev_items * 1.5:
            suffix = "expanded"
        elif curr_items < prev_items * 0.7:
            suffix = "subset"
        elif curr_datasets != prev_datasets:
            suffix = "modified"
        else:
            suffix = "adapted"

        safe_name = f"{original_output_name}_{suffix}"

    else:  # "auto" strategy (default)
        # Intelligent naming based on changes detected
        timestamp = datetime.now().strftime("%m%d_%H%M")
        safe_name = f"{original_output_name}_dataset_change_{timestamp}"

    return safe_name


def smart_resume_validation_with_protection(
    dataset_group: "DatasetGroup",
    resume_dir: str,
    current_output_name: str,
    auto_protect_checkpoints: bool = True,
    naming_strategy: str = "auto",
) -> tuple[bool, Optional[dict], Optional[str]]:
    """
    Smart resume validation with automatic checkpoint protection.

    This is the enhanced version that automatically protects against checkpoint overwrites.

    Args:
        dataset_group: The dataset group to validate
        resume_dir: Directory containing the resume state
        current_output_name: Current output name from config
        auto_protect_checkpoints: Whether to automatically rename output when datasets change
        naming_strategy: Strategy for generating safe names ("auto", "timestamp", "descriptive")

    Returns:
        Tuple of (can_continue, adjustment_info, safe_output_name)
        - can_continue: Whether training can continue safely
        - adjustment_info: Information about any adjustments made (or None)
        - safe_output_name: Protected output name (or None if no change needed)
    """
    is_valid, dataset_changed, expected_info = validate_dataset_for_resume(
        dataset_group=dataset_group, resume_dir=resume_dir, auto_handle_changes=True
    )

    if not is_valid:
        return False, None, None

    if not dataset_changed:
        # Same dataset - no protection needed
        logger.info("📝 Output name unchanged - same dataset detected")
        return True, None, None

    # Dataset changed - apply protection if enabled
    if auto_protect_checkpoints:
        safe_output_name = generate_safe_output_name(
            current_output_name, expected_info, dataset_group, naming_strategy  # type: ignore
        )

        logger.info("🛡️  AUTOMATIC CHECKPOINT PROTECTION ACTIVATED:")
        logger.info(f"   Dataset change detected - protecting existing checkpoints")
        logger.info(f"   Original output name: '{current_output_name}'")
        logger.info(f"   Protected output name: '{safe_output_name}'")
        logger.info("")
        logger.info("📁 This ensures:")
        logger.info("   ✅ Your original checkpoints remain safe and untouched")
        logger.info("   ✅ New training creates separate checkpoint files")
        logger.info("   ✅ You can always go back to your original LoRA")
        logger.info("   ✅ Clear separation between different training phases")
        logger.info("")
        logger.info("💡 If you want to use the original name, manually specify:")
        logger.info(f"   --output_name '{current_output_name}' (with caution)")

        # Prepare adjustment info
        adjustment_info = {
            "dataset_changed": True,
            "expected_info": expected_info,
            "requires_adjustment": True,
            "output_name_changed": True,
            "original_output_name": current_output_name,
            "safe_output_name": safe_output_name,
        }

        return True, adjustment_info, safe_output_name

    else:
        # Protection disabled - warn user about risks
        logger.warning("⚠️  CHECKPOINT PROTECTION DISABLED:")
        logger.warning(f"   Dataset change detected but auto-protection is off")
        logger.warning(f"   Risk: New checkpoints may overwrite existing ones")
        logger.warning(f"   Original output name will be used: '{current_output_name}'")
        logger.warning("")
        logger.warning("💡 To enable automatic protection:")
        logger.warning("   Set auto_protect_checkpoints=True in your training config")

        adjustment_info = {
            "dataset_changed": True,
            "expected_info": expected_info,
            "requires_adjustment": True,
            "output_name_changed": False,
            "protection_disabled": True,
        }

        return True, adjustment_info, None


def apply_smart_resume_adjustments_with_protection(
    adjustment_info: dict, current_global_step: int, dataset_group: "DatasetGroup"
) -> tuple[int, int, dict]:
    """
    Apply smart adjustments for resumed training with dataset changes and checkpoint protection.

    Args:
        adjustment_info: Information returned from smart_resume_validation_with_protection
        current_global_step: Current global step from checkpoint
        dataset_group: Current dataset group

    Returns:
        Tuple of (adjusted_global_step, epoch_start, protection_info)
    """
    if not adjustment_info or not adjustment_info.get("requires_adjustment"):
        return current_global_step, 0, {}

    expected_info = adjustment_info["expected_info"]

    # Apply automatic training state adjustments
    adjusted_step, epoch_start, reset_applied = handle_dataset_change_for_resume_auto(
        expected_info, dataset_group, current_global_step
    )

    protection_info = {
        "training_state_reset": reset_applied,
        "adjusted_global_step": adjusted_step,
        "epoch_start": epoch_start,
        "output_name_changed": adjustment_info.get("output_name_changed", False),
        "safe_output_name": adjustment_info.get("safe_output_name"),
        "original_output_name": adjustment_info.get("original_output_name"),
    }

    if adjustment_info.get("output_name_changed"):
        logger.info("🎯 TRAINING ADJUSTMENTS SUMMARY:")
        logger.info(f"   ✅ Global step: {current_global_step} → {adjusted_step}")
        logger.info(f"   ✅ Epoch start: {epoch_start}")
        logger.info(
            f"   ✅ Output name: {adjustment_info['original_output_name']} → {adjustment_info['safe_output_name']}"
        )
        logger.info(f"   ✅ LoRA weights: Preserved from checkpoint")
        logger.info(
            f"   ✅ Training state: {'Reset for new dataset' if reset_applied else 'Adjusted'}"
        )

    return adjusted_step, epoch_start, protection_info


def smart_resume_validation(
    dataset_group: "DatasetGroup",
    resume_dir: str,
    current_output_name: str = None,  # type: ignore
    auto_protect_checkpoints: bool = True,
) -> tuple[bool, Optional[dict], Optional[str]]:
    """
    Simple smart resume validation with automatic checkpoint protection.

    This is the main function you should use - it automatically protects your checkpoints!

    Args:
        dataset_group: The dataset group to validate
        resume_dir: Directory containing the resume state
        current_output_name: Current output name from config (optional, for protection)
        auto_protect_checkpoints: Whether to automatically rename output when datasets change

    Returns:
        Tuple of (can_continue, adjustment_info, safe_output_name)
        - can_continue: Whether training can continue safely
        - adjustment_info: Information about any adjustments made (or None)
        - safe_output_name: Protected output name (or None if no change needed)
    """
    if current_output_name is None:
        # Backward compatibility - no protection
        is_valid, dataset_changed, expected_info = validate_dataset_for_resume(
            dataset_group=dataset_group, resume_dir=resume_dir, auto_handle_changes=True
        )

        if not is_valid:
            return False, None, None

        if not dataset_changed:
            return True, None, None
        else:
            adjustment_info = {
                "dataset_changed": True,
                "expected_info": expected_info,
                "requires_adjustment": True,
                "output_name_changed": False,
            }
            return True, adjustment_info, None
    else:
        # Full protection enabled
        return smart_resume_validation_with_protection(
            dataset_group=dataset_group,
            resume_dir=resume_dir,
            current_output_name=current_output_name,
            auto_protect_checkpoints=auto_protect_checkpoints,
            naming_strategy="auto",
        )
