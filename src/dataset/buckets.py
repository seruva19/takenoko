import math
import random
from typing import Any, Tuple
import logging
import os
import numpy as np
import torch


from dataset.item_info import ItemInfo
from common.logger import get_logger
from memory.safetensors_loader import load_file


logger = get_logger(__name__, level=logging.INFO)


def divisible_by(num: int, divisor: int) -> int:
    return num - num % divisor


class BucketSelector:
    # TODO: REFACTOR - Hardcoded constant should be configurable per architecture
    RESOLUTION_STEPS_WAN_2 = 16

    def __init__(
        self,
        resolution: Tuple[int, int],
        enable_bucket: bool = True,
        no_upscale: bool = False,
    ):
        # TODO: REFACTOR - Complex bucket calculation logic should be extracted to separate methods
        self.resolution = resolution
        self.bucket_area = resolution[0] * resolution[1]

        self.reso_steps = BucketSelector.RESOLUTION_STEPS_WAN_2

        if not enable_bucket:
            # only define one bucket
            self.bucket_resolutions = [resolution]
            self.no_upscale = False
        else:
            # prepare bucket resolution
            self.no_upscale = no_upscale
            sqrt_size = int(math.sqrt(self.bucket_area))
            min_size = divisible_by(sqrt_size // 2, self.reso_steps)
            self.bucket_resolutions = []
            for w in range(min_size, sqrt_size + self.reso_steps, self.reso_steps):
                h = divisible_by(self.bucket_area // w, self.reso_steps)
                self.bucket_resolutions.append((w, h))
                self.bucket_resolutions.append((h, w))

            self.bucket_resolutions = list(set(self.bucket_resolutions))
            self.bucket_resolutions.sort()

        # calculate aspect ratio to find the nearest resolution
        self.aspect_ratios = np.array([w / h for w, h in self.bucket_resolutions])

    def get_bucket_resolution(self, image_size: tuple[int, int]) -> tuple[int, int]:
        """
        return the bucket resolution for the given image size, (width, height)
        """
        area = image_size[0] * image_size[1]
        if self.no_upscale and area <= self.bucket_area:
            w, h = image_size
            w = divisible_by(w, self.reso_steps)
            h = divisible_by(h, self.reso_steps)
            return w, h

        aspect_ratio = image_size[0] / image_size[1]
        ar_errors = self.aspect_ratios - aspect_ratio
        bucket_id = np.abs(ar_errors).argmin()
        return self.bucket_resolutions[bucket_id]


class BucketBatchManager:
    # TODO: REFACTOR - Type hints are too generic (tuple[Any])
    # Should use proper type hints like tuple[int, int] or tuple[int, int, int]

    def __init__(
        self,
        bucketed_item_info: dict[Any, list[ItemInfo]],
        batch_size: int,
        prior_loss_weight: float = 1.0,
    ):
        self.batch_size = batch_size
        self.buckets = bucketed_item_info
        self.prior_loss_weight = prior_loss_weight
        self.bucket_resos = list(self.buckets.keys())
        self.bucket_resos.sort()
        # Optional per-epoch timestep bucketing
        self.num_timestep_buckets: int | None = None
        self._timestep_pool_batches: list[list[float]] | None = None

        # indices for enumerating batches. each batch is reso + batch_idx. reso is (width, height) or (width, height, frames)
        self.bucket_batch_indices: list[tuple[tuple[Any], int]] = []
        for bucket_reso in self.bucket_resos:
            bucket = self.buckets[bucket_reso]
            num_batches = math.ceil(len(bucket) / self.batch_size)
            for i in range(num_batches):
                self.bucket_batch_indices.append((bucket_reso, i))

        # do no shuffle here to avoid multiple datasets have different order
        # self.shuffle()

    def show_bucket_info(self):
        for bucket_reso in self.bucket_resos:
            bucket = self.buckets[bucket_reso]
            logger.info(f"bucket: {bucket_reso}, count: {len(bucket)}")

        logger.info(f"total batches: {len(self)}")

    def shuffle(self):
        # shuffle each bucket
        for bucket in self.buckets.values():
            random.shuffle(bucket)

        # shuffle the order of batches
        random.shuffle(self.bucket_batch_indices)

        # Prepare per-epoch timestep pool if enabled
        self._prepare_timestep_pool_internal()

    def set_num_timestep_buckets(self, num_timestep_buckets: int | None) -> None:
        self.num_timestep_buckets = (
            int(num_timestep_buckets) if num_timestep_buckets is not None else None
        )


    def prepare_timestep_pool(self) -> None:
        """Create/refresh the per-epoch timestep pool without shuffling batches."""
        self._prepare_timestep_pool_internal()

    def _prepare_timestep_pool_internal(self) -> None:
        self._timestep_pool_batches = None
        if self.num_timestep_buckets is not None and self.num_timestep_buckets > 1:
            try:
                num_batches = len(self.bucket_batch_indices)
                total_timesteps_needed = num_batches * self.batch_size
                samples_per_bucket = math.ceil(
                    total_timesteps_needed / int(self.num_timestep_buckets)
                )
                all_timesteps: list[float] = []
                for i in range(int(self.num_timestep_buckets)):
                    min_t = i / float(self.num_timestep_buckets)
                    max_t = (i + 1) / float(self.num_timestep_buckets)
                    for _ in range(samples_per_bucket):
                        all_timesteps.append(random.uniform(min_t, max_t))
                random.shuffle(all_timesteps)
                # trim
                all_timesteps = all_timesteps[:total_timesteps_needed]
                # chunk into batches
                self._timestep_pool_batches = []
                for i in range(num_batches):
                    s = i * self.batch_size
                    e = s + self.batch_size
                    self._timestep_pool_batches.append(all_timesteps[s:e])
            except Exception:
                # Non-fatal: fallback to no preset timesteps for this epoch
                self._timestep_pool_batches = None

    def __len__(self):
        return len(self.bucket_batch_indices)

    def __getitem__(self, idx):
        # TODO: REFACTOR - This method is too complex and handles multiple responsibilities
        # Consider extracting tensor processing logic into separate methods
        bucket_reso, batch_idx = self.bucket_batch_indices[idx]
        bucket = self.buckets[bucket_reso]
        start = batch_idx * self.batch_size
        end = min(start + self.batch_size, len(bucket))

        batch_tensor_data = {}
        varlen_keys = set()
        weights = []  # Collect weights for the batch
        control_signals = []  # Collect control signals for the batch
        mask_signals = []  # Collect mask signals for the batch
        pixels_list = []  # Collect original (or resized) pixel tensors

        for item_info in bucket[start:end]:
            if item_info.latent_cache_path is not None:
                sd_latent = load_file(item_info.latent_cache_path)
            else:
                sd_latent = {}

            if item_info.text_encoder_output_cache_path is not None:
                sd_te = load_file(item_info.text_encoder_output_cache_path)
            else:
                sd_te = {}

            sd = {**sd_latent, **sd_te}

            # Add weight to the batch
            # This is the key change: determine the weight for this specific item
            loss_weight = self.prior_loss_weight if item_info.is_reg else 1.0
            weights.append(loss_weight)

            # Load control signal if available
            if (
                hasattr(item_info, "control_content")
                and item_info.control_content is not None
            ):
                # Convert control content to tensor
                control_tensor = torch.from_numpy(item_info.control_content).float()
                control_signals.append(control_tensor)
            else:
                # Check if control cache file exists
                if item_info.latent_cache_path is not None:
                    control_cache_path = item_info.latent_cache_path.replace(
                        ".safetensors", "_control.safetensors"
                    )
                    if os.path.exists(control_cache_path):
                        try:
                            control_sd = load_file(control_cache_path)
                            # Find the control latent key
                            control_key = None
                            for key in control_sd.keys():
                                if key.startswith("latents_"):
                                    control_key = key
                                    break
                            if control_key:
                                control_signals.append(control_sd[control_key])
                            else:
                                control_signals.append(None)
                        except Exception as e:
                            logger.warning(
                                f"Failed to load control cache {control_cache_path}: {e}"
                            )
                            control_signals.append(None)
                    else:
                        control_signals.append(None)
                else:
                    control_signals.append(None)

            # Load mask signal if available
            if (
                hasattr(item_info, "mask_content")
                and item_info.mask_content is not None
            ):
                # Convert mask content to tensor
                mask_tensor = torch.from_numpy(item_info.mask_content).float()
                mask_signals.append(mask_tensor)
            else:
                # Check if mask cache file exists
                if item_info.latent_cache_path is not None:
                    mask_cache_path = item_info.latent_cache_path.replace(
                        ".safetensors", "_mask.safetensors"
                    )
                    if os.path.exists(mask_cache_path):
                        try:
                            mask_sd = load_file(mask_cache_path)
                            # Find the mask latent key
                            mask_key = None
                            for key in mask_sd.keys():
                                if key.startswith("latents_"):
                                    mask_key = key
                                    break
                            if mask_key:
                                mask_signals.append(mask_sd[mask_key])
                            else:
                                mask_signals.append(None)
                        except Exception as e:
                            logger.warning(
                                f"Failed to load mask cache {mask_cache_path}: {e}"
                            )
                            mask_signals.append(None)
                    else:
                        mask_signals.append(None)
                else:
                    mask_signals.append(None)

            # TODO: REFACTOR - Complex key processing logic should be extracted to a separate method
            # This logic is hard to understand and maintain
            for key in sd.keys():
                is_varlen_key = key.startswith("varlen_")  # varlen keys are not stacked
                content_key = key

                if is_varlen_key:
                    content_key = content_key.replace("varlen_", "")

                # We need a more robust way to parse keys now.
                # Example keys: "t5_bf16", "t5_preservation_bf16"
                if "preservation" in content_key:
                    # This is a preservation embedding
                    parts = content_key.split("_")
                    # Should be [text_encoder_type, "preservation", dtype]
                    # We will name it "t5_preservation" in the batch
                    if len(parts) == 3:
                        content_key = f"{parts[0]}_preservation"

                elif content_key.endswith("_mask"):
                    pass
                else:
                    content_key = content_key.rsplit("_", 1)[0]  # remove dtype
                    if content_key.startswith("latents_"):
                        content_key = content_key.rsplit("_", 1)[0]  # remove FxHxW

                if content_key not in batch_tensor_data:
                    batch_tensor_data[content_key] = []
                batch_tensor_data[content_key].append(sd[key])

                if is_varlen_key:
                    varlen_keys.add(content_key)

            # Collect pixels for on-the-fly control processing if available
            if item_info.content is not None:
                # item_info.content is an np.ndarray resized to bucket resolution
                arr = item_info.content  # (H,W,C) for images, (F,H,W,C) for videos
                if arr.ndim == 3:  # single image -> create dummy frame dim F=1
                    tensor = (
                        torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(1)
                    )  # C,F,H,W
                elif arr.ndim == 4:  # video
                    tensor = torch.from_numpy(arr).permute(3, 0, 1, 2)  # C,F,H,W
                else:
                    raise ValueError(f"Unsupported content shape: {arr.shape}")

                # Normalize to [-1,1] like reference implementation
                tensor = tensor.float().div(127.5).sub(1.0)
                pixels_list.append(tensor)

        for key in batch_tensor_data.keys():
            if key not in varlen_keys:
                batch_tensor_data[key] = torch.stack(batch_tensor_data[key])

        # Add weights to the batch tensor data
        batch_tensor_data["weight"] = torch.tensor(weights, dtype=torch.float32)

        # Add control signals to batch if any are available
        if any(cs is not None for cs in control_signals):
            # Determine reference tensor to preserve shape/dtype
            ref_cs = next((cs for cs in control_signals if cs is not None), None)
            if ref_cs is not None:
                stacked_control_signals: list[torch.Tensor] = []
                for cs in control_signals:
                    if cs is None:
                        stacked_control_signals.append(torch.zeros_like(ref_cs))
                    else:
                        stacked_control_signals.append(cs)
                batch_tensor_data["control_signal"] = torch.stack(
                    stacked_control_signals
                )

        # Add mask signals to batch if any are available
        if any(ms is not None for ms in mask_signals):
            ref_ms = next((ms for ms in mask_signals if ms is not None), None)
            if ref_ms is not None:
                stacked_mask_signals: list[torch.Tensor] = []
                for ms in mask_signals:
                    if ms is None:
                        stacked_mask_signals.append(torch.zeros_like(ref_ms))
                    else:
                        stacked_mask_signals.append(ms)
                batch_tensor_data["mask_signal"] = torch.stack(stacked_mask_signals)

        # Add pixels tensor if collected - format to match reference implementation
        if pixels_list:
            # Reference expects individual CFHW tensors, not stacked tensor
            # Our pixels_list contains CFHW tensors, which is correct
            # Store as individual tensors to match reference collate_batch format
            batch_tensor_data["pixels"] = pixels_list

        # Add item_info to batch for control video caching
        batch_tensor_data["item_info"] = bucket[start:end]

        # Attach per-batch timesteps if prepared
        if self._timestep_pool_batches is not None:
            try:
                # Respect possible last partial batch size
                timesteps = self._timestep_pool_batches[idx][: end - start]
                batch_tensor_data["timesteps"] = timesteps
            except Exception:
                batch_tensor_data["timesteps"] = None
        else:
            batch_tensor_data["timesteps"] = None

        return batch_tensor_data
