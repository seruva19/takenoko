import argparse

import os
import shutil

import accelerate

from datetime import datetime, timedelta
import gc
import argparse
import math
import re
import time
import json
from typing import Dict, Optional
import accelerate
from packaging.version import Version

import toml

import torch
from accelerate.utils import TorchDynamoPlugin, DynamoBackend
from accelerate import (
    Accelerator,
    InitProcessGroupKwargs,
    DistributedDataParallelKwargs,
)


import logging
from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)

# Guard to emit certain warnings only once per run (per source)
_warned_missing_timesteps_sources: set[str] = set()


TAKENOKO_METADATA_KEY_BASE_MODEL_VERSION = "takenoko_base_model_version"
TAKENOKO_METADATA_KEY_NETWORK_MODULE = "takenoko_network_module"
TAKENOKO_METADATA_KEY_NETWORK_DIM = "takenoko_network_dim"
TAKENOKO_METADATA_KEY_NETWORK_ALPHA = "takenoko_network_alpha"
TAKENOKO_METADATA_KEY_NETWORK_ARGS = "takenoko_network_args"

TAKENOKO_METADATA_MINIMUM_KEYS = [
    TAKENOKO_METADATA_KEY_BASE_MODEL_VERSION,
    TAKENOKO_METADATA_KEY_NETWORK_MODULE,
    TAKENOKO_METADATA_KEY_NETWORK_DIM,
    TAKENOKO_METADATA_KEY_NETWORK_ALPHA,
    TAKENOKO_METADATA_KEY_NETWORK_ARGS,
]


# checkpoint
EPOCH_STATE_NAME = "{}-epoch{:06d}-state"
EPOCH_FILE_NAME = "{}-epoch{:06d}"
EPOCH_DIFFUSERS_DIR_NAME = "{}-epoch{:06d}"
LAST_STATE_NAME = "{}-state"
STEP_STATE_NAME = "{}-step{:08d}-state"
STEP_FILE_NAME = "{}-step{:08d}"
STEP_DIFFUSERS_DIR_NAME = "{}-step{:08d}"


def get_time_flag():
    return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S-%f")[:-3]


def get_sanitized_config_or_none(args: argparse.Namespace):
    # if `--log_config` is enabled, return args for logging. if not, return None.
    # when `--log_config is enabled, filter out sensitive values from args

    if not args.log_config:
        return None

    sensitive_args = []
    sensitive_path_args = [
        "dit",
        "vae",
        "text_encoder1",
        "text_encoder2",
        "image_encoder",
        "base_weights",
        "network_weights",
        "output_dir",
        "logging_dir",
    ]
    filtered_args = {}
    for k, v in vars(args).items():
        # filter out sensitive values and convert to string if necessary
        if k not in sensitive_args + sensitive_path_args:
            # Accelerate values need to have type `bool`,`str`, `float`, `int`, or `None`.
            if (
                v is None
                or isinstance(v, bool)
                or isinstance(v, str)
                or isinstance(v, float)
                or isinstance(v, int)
            ):
                filtered_args[k] = v
            # accelerate does not support lists
            elif isinstance(v, list):
                filtered_args[k] = f"{v}"
            # accelerate does not support objects
            elif isinstance(v, object):
                filtered_args[k] = f"{v}"

    return filtered_args


class LossRecorder:
    def __init__(self):
        self.loss_list: list[float] = []
        self.loss_total: float = 0.0

    def add(self, *, epoch: int, step: int, loss: float) -> None:
        if epoch == 0:
            self.loss_list.append(loss)
        else:
            while len(self.loss_list) <= step:
                self.loss_list.append(0.0)
            self.loss_total -= self.loss_list[step]
            self.loss_list[step] = loss
        self.loss_total += loss

    @property
    def moving_average(self) -> float:
        return self.loss_total / len(self.loss_list)


def get_epoch_ckpt_name(model_name, epoch_no: int):
    return EPOCH_FILE_NAME.format(model_name, epoch_no) + ".safetensors"


def get_step_ckpt_name(model_name, step_no: int):
    return STEP_FILE_NAME.format(model_name, step_no) + ".safetensors"


def get_last_ckpt_name(model_name):
    return model_name + ".safetensors"


def get_remove_epoch_no(args: argparse.Namespace, epoch_no: int):
    if args.save_last_n_epochs is None:
        return None

    remove_epoch_no = epoch_no - args.save_every_n_epochs * args.save_last_n_epochs
    if remove_epoch_no < 0:
        return None
    return remove_epoch_no


def get_remove_step_no(args: argparse.Namespace, step_no: int):
    if args.save_last_n_steps is None:
        return None

    # calculate the step number to remove from the last_n_steps and save_every_n_steps
    # e.g. if save_every_n_steps=10, save_last_n_steps=30, at step 50, keep 30 steps and remove step 10
    remove_step_no = step_no - args.save_last_n_steps - 1
    remove_step_no = remove_step_no - (remove_step_no % args.save_every_n_steps)
    if remove_step_no < 0:
        return None
    return remove_step_no


def save_and_remove_state_on_epoch_end(
    args: argparse.Namespace, accelerator: accelerate.Accelerator, epoch_no: int
):
    model_name = args.output_name

    logger.info(f"💾 Saving state at epoch {epoch_no}")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(
        args.output_dir, EPOCH_STATE_NAME.format(model_name, epoch_no)
    )
    accelerator.save_state(state_dir)

    # Save original config file to state directory
    save_original_config_to_state_dir(state_dir, args)

    last_n_epochs = (
        args.save_last_n_epochs_state
        if args.save_last_n_epochs_state
        else args.save_last_n_epochs
    )
    if last_n_epochs is not None:
        remove_epoch_no = epoch_no - args.save_every_n_epochs * last_n_epochs
        state_dir_old = os.path.join(
            args.output_dir, EPOCH_STATE_NAME.format(model_name, remove_epoch_no)
        )
        if os.path.exists(state_dir_old):
            logger.info(f"removing old state: {state_dir_old}")
            shutil.rmtree(state_dir_old)


def save_original_config_to_state_dir(state_dir: str, args: argparse.Namespace) -> None:
    """Save the original config file content to the state directory for reproducibility."""
    try:
        if hasattr(args, "config_content") and args.config_content is not None:
            config_filename = "original_config.toml"
            if hasattr(args, "config_file") and args.config_file is not None:
                # Extract just the filename from the path
                import os

                config_filename = os.path.basename(args.config_file)
                if not config_filename.endswith(".toml"):
                    config_filename += ".toml"

            config_file_path = os.path.join(state_dir, config_filename)
            with open(config_file_path, "w", encoding="utf-8") as f:
                f.write(args.config_content)
            logger.info(f"Saved original config to: {config_file_path}")
        else:
            logger.debug("No original config content available to save")
    except Exception as e:
        logger.warning(f"Failed to save original config to state directory: {e}")


def save_and_remove_state_stepwise(
    args: argparse.Namespace, accelerator: accelerate.Accelerator, step_no: int
):
    model_name = args.output_name

    logger.info(f"💾 Saving state at step {step_no}")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(
        args.output_dir, STEP_STATE_NAME.format(model_name, step_no)
    )
    accelerator.save_state(state_dir)

    # Save original config file to state directory
    save_original_config_to_state_dir(state_dir, args)

    # Save step number to step.txt
    step_file = os.path.join(state_dir, "step.txt")
    try:
        with open(step_file, "w") as f:
            f.write(str(step_no))
    except Exception as e:
        logger.warning(f"Failed to write step.txt: {e}")

    last_n_steps = (
        args.save_last_n_steps_state
        if args.save_last_n_steps_state
        else args.save_last_n_steps
    )
    if last_n_steps is not None:
        remove_step_no = step_no - last_n_steps - 1
        remove_step_no = remove_step_no - (remove_step_no % args.save_every_n_steps)

        if remove_step_no > 0:
            state_dir_old = os.path.join(
                args.output_dir, STEP_STATE_NAME.format(model_name, remove_step_no)
            )
            if os.path.exists(state_dir_old):
                logger.info(f"removing old state: {state_dir_old}")
                shutil.rmtree(state_dir_old)


def should_save_state_at_epoch(args: argparse.Namespace, epoch: int) -> bool:
    """Check if state should be saved at this epoch based on save_state_every_n_epochs."""
    if not getattr(args, "save_state", True):
        return False

    save_state_every_n_epochs = getattr(args, "save_state_every_n_epochs", None)
    if save_state_every_n_epochs is None:
        return False

    return (epoch % save_state_every_n_epochs) == 0


def should_save_state_at_step(args: argparse.Namespace, step: int) -> bool:
    """Check if state should be saved at this step based on save_state_every_n_steps."""
    if not getattr(args, "save_state", True):
        return False

    save_state_every_n_steps = getattr(args, "save_state_every_n_steps", None)
    if save_state_every_n_steps is None:
        return False

    return (step % save_state_every_n_steps) == 0


def save_state_only_at_epoch(
    args: argparse.Namespace, accelerator: accelerate.Accelerator, epoch_no: int
) -> None:
    """Save only training state (not checkpoint) at specified epoch."""
    if not accelerator.is_main_process:
        return

    model_name = args.output_name
    logger.info(f"💾 Saving state-only at epoch {epoch_no}")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(
        args.output_dir, EPOCH_STATE_NAME.format(model_name, epoch_no)
    )
    accelerator.save_state(state_dir)

    # Save original config file to state directory
    save_original_config_to_state_dir(state_dir, args)

    # Clean up old state-only saves if configured
    last_n_epochs = getattr(args, "save_last_n_epochs_state", None)
    save_state_every_n_epochs = getattr(args, "save_state_every_n_epochs", None)

    if last_n_epochs is not None and save_state_every_n_epochs is not None:
        remove_epoch_no = epoch_no - save_state_every_n_epochs * last_n_epochs
        state_dir_old = os.path.join(
            args.output_dir, EPOCH_STATE_NAME.format(model_name, remove_epoch_no)
        )
        if os.path.exists(state_dir_old):
            logger.info(f"removing old state-only: {state_dir_old}")
            shutil.rmtree(state_dir_old)


def save_state_only_at_step(
    args: argparse.Namespace, accelerator: accelerate.Accelerator, step_no: int
) -> None:
    """Save only training state (not checkpoint) at specified step."""
    if not accelerator.is_main_process:
        return

    model_name = args.output_name
    logger.info(f"💾 Saving state-only at step {step_no}")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(
        args.output_dir, STEP_STATE_NAME.format(model_name, step_no)
    )
    accelerator.save_state(state_dir)

    # Save original config file to state directory
    save_original_config_to_state_dir(state_dir, args)

    # Save step number to step.txt
    step_file = os.path.join(state_dir, "step.txt")
    try:
        with open(step_file, "w") as f:
            f.write(str(step_no))
    except Exception as e:
        logger.warning(f"Failed to write step.txt: {e}")

    # Clean up old state-only saves if configured
    last_n_steps = getattr(args, "save_last_n_steps_state", None)
    save_state_every_n_steps = getattr(args, "save_state_every_n_steps", None)

    if last_n_steps is not None and save_state_every_n_steps is not None:
        remove_step_no = step_no - (save_state_every_n_steps * last_n_steps)
        if remove_step_no > 0:
            state_dir_old = os.path.join(
                args.output_dir, STEP_STATE_NAME.format(model_name, remove_step_no)
            )
            if os.path.exists(state_dir_old):
                logger.info(f"removing old state-only: {state_dir_old}")
                shutil.rmtree(state_dir_old)


def save_state_on_train_end(
    args: argparse.Namespace, accelerator: accelerate.Accelerator
):
    model_name = args.output_name

    logger.info("💾 Saving final state")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(args.output_dir, LAST_STATE_NAME.format(model_name))
    accelerator.save_state(state_dir)

    # Save original config file to state directory
    save_original_config_to_state_dir(state_dir, args)


def clean_memory_on_device(device: torch.device):
    r"""
    Clean memory on the specified device, will be called from training scripts.
    """
    gc.collect()

    # device may "cuda" or "cuda:0", so we need to check the type of device
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "xpu":
        torch.xpu.empty_cache()
    if device.type == "mps":
        torch.mps.empty_cache()


# for collate_fn: epoch and step is multiprocessing.Value
class collator_class:
    def __init__(self, epoch, step, dataset):
        self.current_epoch = epoch
        self.current_step = step
        self.dataset = (
            dataset  # not used if worker_info is not None, in case of multiprocessing
        )

    def __call__(self, examples):
        worker_info = torch.utils.data.get_worker_info()
        # worker_info is None in the main process
        if worker_info is not None:
            dataset = worker_info.dataset
            context_reason = "dataloader_worker"
        else:
            dataset = self.dataset
            context_reason = "main_process"

        # Conservative epoch setting with explicit shuffle control
        try:
            # Build reason string for better logging context
            reason = f"collator_{context_reason}"

            # CRITICAL: Always disable shuffling for collator calls
            # Collator calls are synchronization operations, not training progression
            # Shuffling should only happen during genuine epoch progression in training loops
            dataset.set_current_epoch(  # type: ignore
                self.current_epoch.value,
                force_shuffle=False,  # Explicitly disable shuffling for all collator calls
                reason=reason,
            )
        except (AttributeError, TypeError) as e:
            # Fallback for datasets that don't support the new signature
            try:
                dataset.set_current_epoch(self.current_epoch.value)  # type: ignore
            except AttributeError:
                pass  # Dataset doesn't have set_current_epoch method

        try:
            dataset.set_current_step(self.current_step.value)  # type: ignore
        except AttributeError:
            pass  # Dataset doesn't have set_current_step method

        return examples[0]


def prepare_accelerator(args: argparse.Namespace) -> Accelerator:
    """
    DeepSpeed is not supported in this script currently.
    """
    if args.logging_dir is None:
        logging_dir = None
    else:
        log_prefix = "" if args.log_prefix is None else args.log_prefix
        logging_dir = (
            args.logging_dir
            + "/"
            + log_prefix
            + time.strftime("%Y%m%d%H%M%S", time.localtime())
        )

    if args.log_with is None:
        if logging_dir is not None:
            log_with = "tensorboard"
        else:
            log_with = None
    else:
        requested = str(args.log_with).lower()
        if not requested in ["tensorboard"]:
            logger.warning(
                "only tensorboard logging is supported; using TensorBoard instead"
            )
            log_with = "tensorboard" if logging_dir is not None else None
        else:
            log_with = requested
        if log_with == "tensorboard" and logging_dir is None:
            raise ValueError(
                "logging_dir must be specified when using TensorBoard logging."
            )

    kwargs_handlers = [
        (
            InitProcessGroupKwargs(
                backend=(
                    "gloo"
                    if os.name == "nt" or not torch.cuda.is_available()
                    else "nccl"
                ),
                init_method=(
                    "env://?use_libuv=False"
                    if os.name == "nt"
                    and Version(torch.__version__) >= Version("2.4.0")
                    else None
                ),
                timeout=(
                    timedelta(minutes=args.ddp_timeout) if args.ddp_timeout else None
                ),
            )
            if torch.cuda.device_count() > 1
            else None
        ),
        (
            DistributedDataParallelKwargs(
                gradient_as_bucket_view=args.ddp_gradient_as_bucket_view,
                static_graph=args.ddp_static_graph,
            )
            if args.ddp_gradient_as_bucket_view or args.ddp_static_graph
            else None
        ),
    ]
    kwargs_handlers = [i for i in kwargs_handlers if i is not None]

    dynamo_plugin = None
    if args.dynamo_backend.upper() != "NO":
        dynamo_plugin = TorchDynamoPlugin(
            backend=DynamoBackend(args.dynamo_backend.upper()),
            mode=args.dynamo_mode,
            fullgraph=args.dynamo_fullgraph,
            dynamic=args.dynamo_dynamic,
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=log_with,
        project_dir=logging_dir,
        dynamo_plugin=dynamo_plugin,
        kwargs_handlers=kwargs_handlers,
    )
    return accelerator


def line_to_prompt_dict(line: str) -> dict:
    # subset of gen_img_diffusers
    prompt_args = line.split(" --")
    prompt_dict = {}
    prompt_dict["prompt"] = prompt_args[0]

    for parg in prompt_args:
        try:
            m = re.match(r"w (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["width"] = int(m.group(1))
                continue

            m = re.match(r"h (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["height"] = int(m.group(1))
                continue

            m = re.match(r"f (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["frame_count"] = int(m.group(1))
                continue

            m = re.match(r"d (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["seed"] = int(m.group(1))
                continue

            m = re.match(r"s (\d+)", parg, re.IGNORECASE)
            if m:  # steps
                prompt_dict["sample_steps"] = max(1, min(1000, int(m.group(1))))
                continue

            m = re.match(r"g ([\d\.]+)", parg, re.IGNORECASE)
            if m:  # scale
                prompt_dict["guidance_scale"] = float(m.group(1))
                continue

            m = re.match(r"fs ([\d\.]+)", parg, re.IGNORECASE)
            if m:  # scale
                prompt_dict["discrete_flow_shift"] = float(m.group(1))
                continue

            m = re.match(r"l ([\d\.]+)", parg, re.IGNORECASE)
            if m:  # scale
                prompt_dict["cfg_scale"] = float(m.group(1))
                continue

            m = re.match(r"n (.+)", parg, re.IGNORECASE)
            if m:  # negative prompt
                prompt_dict["negative_prompt"] = m.group(1)
                continue

            m = re.match(r"i (.+)", parg, re.IGNORECASE)
            if m:  # negative prompt
                prompt_dict["image_path"] = m.group(1)
                continue

            m = re.match(r"cn (.+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["control_video_path"] = m.group(1)
                continue

            m = re.match(r"ci (.+)", parg, re.IGNORECASE)
            if m:
                # can be multiple control images
                control_image_path = m.group(1)
                if "control_image_path" not in prompt_dict:
                    prompt_dict["control_image_path"] = []
                prompt_dict["control_image_path"].append(control_image_path)
                continue

            m = re.match(r"of (.+)", parg, re.IGNORECASE)
            if m:  # output folder
                prompt_dict["one_frame"] = m.group(1)
                continue

        except ValueError as ex:
            logger.error(f"Exception occurred while parsing prompt argument: {parg}")
            logger.error(ex)

    return prompt_dict


def load_prompts(prompt_file: str) -> list[Dict]:
    # Validate input
    if not prompt_file or not prompt_file.strip():
        raise ValueError("prompt_file cannot be empty or None")

    logger.info(f"load_prompts called with prompt_file='{prompt_file}'")

    # read prompts
    if prompt_file.endswith(".txt"):
        logger.info(f"Loading from .txt file")
        with open(prompt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        prompts = [
            line.strip() for line in lines if len(line.strip()) > 0 and line[0] != "#"
        ]
    elif prompt_file.endswith(".toml"):
        logger.info(f"Loading from .toml file")
        with open(prompt_file, "r", encoding="utf-8") as f:
            data = toml.load(f)

        # Handle new TOML structure with sample_prompts array
        if "sample_prompts" in data:
            logger.info(
                f"Found sample_prompts in TOML, count: {len(data['sample_prompts'])}"
            )
            prompts = []
            for prompt_dict in data["sample_prompts"]:
                # Convert the new structure to the expected format
                converted_dict = {}
                if "text" in prompt_dict:
                    converted_dict["prompt"] = prompt_dict["text"]
                if "width" in prompt_dict:
                    converted_dict["width"] = prompt_dict["width"]
                if "height" in prompt_dict:
                    converted_dict["height"] = prompt_dict["height"]
                if "frames" in prompt_dict:
                    converted_dict["frame_count"] = prompt_dict["frames"]
                if "seed" in prompt_dict:
                    converted_dict["seed"] = prompt_dict["seed"]
                if "step" in prompt_dict:
                    converted_dict["sample_steps"] = prompt_dict["step"]
                if "control_path" in prompt_dict:
                    converted_dict["control_path"] = prompt_dict["control_path"]
                # Add other fields as needed
                prompts.append(converted_dict)
        else:
            logger.info(f"Using old TOML structure")
            # Handle old TOML structure
            prompts = [
                dict(**data["prompt"], **subset) for subset in data["prompt"]["subset"]
            ]
    elif prompt_file.endswith(".json"):
        logger.info(f"Loading from .json file")
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompts = json.load(f)
    else:
        raise ValueError(
            f"Unsupported file format: {prompt_file}. Supported formats: .txt, .toml, .json"
        )

    logger.info(f"Loaded {len(prompts)} prompts from {prompt_file}")

    # preprocess prompts
    for i in range(len(prompts)):
        prompt_dict = prompts[i]
        if isinstance(prompt_dict, str):
            prompt_dict = line_to_prompt_dict(prompt_dict)
            prompts[i] = prompt_dict  # type: ignore
        assert isinstance(prompt_dict, dict)

        # Adds an enumerator to the dict based on prompt position. Used later to name image files. Also cleanup of extra data in original prompt dict.
        prompt_dict["enum"] = i
        prompt_dict.pop("subset", None)

    logger.info(f"Final processed prompts count: {len(prompts)}")
    return prompts  # type: ignore


def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    logit_mean: float = None,  # type: ignore
    logit_std: float = None,  # type: ignore
    mode_scale: float = None,  # type: ignore
):
    """Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(
            mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu"
        )
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
    return u


def get_sigmas(
    noise_scheduler,
    timesteps,
    device,
    n_dim=4,
    dtype=torch.float32,
    *,
    source: str | None = None,
    timestep_layout: str = "auto",  # one of: "auto", "per_sample" ([B]), "per_frame" ([B, F])
):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)

    # Resolve layout routing
    if timestep_layout not in ("auto", "per_sample", "per_frame"):
        timestep_layout = "auto"

    if timestep_layout == "auto":
        layout = "per_frame" if timesteps.dim() > 1 else "per_sample"
    else:
        layout = timestep_layout

    # Prepare shapes depending on layout
    original_shape = timesteps.shape
    if layout == "per_frame":
        # Expect [B, F]; flatten for indexing
        timesteps_flat = timesteps.flatten()
    else:
        # per_sample: expect [B]
        timesteps_flat = timesteps.view(-1)

    # if sum([(schedule_timesteps == t) for t in timesteps_flat]) < len(timesteps_flat):
    if any([(schedule_timesteps == t).sum() == 0 for t in timesteps_flat]):
        # round to nearest timestep
        global _warned_missing_timesteps_sources
        src = source or "unspecified"
        if src not in _warned_missing_timesteps_sources:
            if source:
                logger.warning(
                    f"🔔 Some timesteps are not present in the noise schedule (source: {source}). Rounding to the nearest available timestep. This is expected when using continuous timesteps; training is not affected."
                )
            else:
                logger.warning(
                    "🔔 Some timesteps are not present in the noise schedule. Rounding to the nearest available timestep. This is expected when using continuous timesteps; training is not affected."
                )
            _warned_missing_timesteps_sources.add(src)
        step_indices = [
            torch.argmin(torch.abs(schedule_timesteps - t)).item()
            for t in timesteps_flat
        ]
    else:
        step_indices = [
            (schedule_timesteps == t).nonzero().item() for t in timesteps_flat
        ]

    sigma = sigmas[step_indices].flatten()

    # Reshape/broadcast according to layout
    if layout == "per_frame":
        # [B*F] -> [B, F]
        sigma = sigma.view(original_shape)
        # Broadcast to [B, 1, F, 1, 1] when n_dim=5 (or analogous for other n_dim)
        # Insert channel axis at dim=1, then expand trailing dims
        if n_dim >= 3:
            if sigma.dim() == 2:
                sigma = sigma.unsqueeze(1)  # [B, 1, F]
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
    else:
        # per_sample: [B] -> [B, 1, 1, ...]
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)

    return sigma


def compute_loss_weighting_for_sd3(
    weighting_scheme: str, noise_scheduler, timesteps, device, dtype
):
    """Computes loss weighting scheme for SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "sigma_sqrt" or weighting_scheme == "cosmap":
        layout = (
            "per_frame"
            if hasattr(timesteps, "dim") and timesteps.dim() > 1
            else "per_sample"
        )
        sigmas = get_sigmas(
            noise_scheduler,
            timesteps,
            device,
            n_dim=5,
            dtype=dtype,
            timestep_layout=layout,
            source="training/weighting",
        )
        if weighting_scheme == "sigma_sqrt":
            weighting = (sigmas**-2.0).float()
        else:
            bot = 1 - 2 * sigmas + 2 * sigmas**2
            weighting = 2 / (math.pi * bot)
    else:
        weighting = None  # torch.ones_like(sigmas)
    return weighting


def should_sample_images(args, steps, epoch=None):
    if steps == 0:
        if not args.sample_at_first:
            return False
        else:
            return True
    else:
        should_sample_by_steps = (
            args.sample_every_n_steps is not None
            and args.sample_every_n_steps > 0
            and steps % args.sample_every_n_steps == 0
        )
        should_sample_by_epochs = (
            args.sample_every_n_epochs is not None
            and args.sample_every_n_epochs > 0
            and epoch is not None
            and epoch % args.sample_every_n_epochs == 0
        )

        if not should_sample_by_steps and not should_sample_by_epochs:
            return False

        return True


# Utility to read step from step.txt in state dir
def read_step_from_state_dir(state_dir: str) -> Optional[int]:
    step_file = os.path.join(state_dir, "step.txt")
    if os.path.exists(step_file):
        try:
            with open(step_file, "r") as f:
                return int(f.read().strip())
        except Exception as e:
            logger.warning(f"Failed to read step.txt: {e}")
    return None


# Utility to save step to step.txt in state dir
def save_step_to_state_dir(state_dir: str, step: int) -> None:
    """Save step number to step.txt for proper checkpoint resuming."""
    os.makedirs(state_dir, exist_ok=True)
    step_file = os.path.join(state_dir, "step.txt")
    try:
        with open(step_file, "w") as f:
            f.write(str(step))
    except Exception as e:
        logger.warning(f"Failed to write step.txt: {e}")
