"""Entry-point wrapper that wires Takenoko config and resources into the RCM core."""

from __future__ import annotations

import argparse
import copy
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, Optional

from accelerate import Accelerator
import torch

from common.logger import get_logger

from core.checkpoint_manager import CheckpointManager
from distillation.rcm_core.tokenizer import TokenizerAssets, RCMTokenizer

from distillation.rcm_core.config_loader import RCMConfig, load_rcm_config
from distillation.rcm_core.runner import run_distillation
from distillation.rcm_dataset_adapter import create_rcm_dataset
from distillation.rcm_model_factory import create_rcm_models
from distillation.rcm_optimizer import build_rcm_optimizer
from distillation.rcm_core.ema import RCMEMAHelper

logger = get_logger(__name__)


def dispatch_rcm_pipeline(
    *,
    args: argparse.Namespace,
    raw_config: Dict[str, Any],
    raw_config_content: str,
    config_path: str,
) -> bool:
    """Main integration point called by ``UnifiedTrainer`` when RCM is enabled."""

    logger.info("Initialising RCM pipeline dispatch from config '%s'", config_path)

    rcm_config = load_rcm_config(args)
    accelerator = _create_accelerator(args, rcm_config)
    hook_handles = _setup_enhancement_hooks(args, accelerator)

    try:
        dataset = create_rcm_dataset(
            args,
            rcm_config,
            raw_config=raw_config,
            config_path=config_path,
            device=accelerator.device,
        )
        teacher, student, teacher_ema, fake_score = create_rcm_models(
            args,
            rcm_config,
            accelerator=accelerator,
            raw_config=raw_config,
            config_path=config_path,
            device=accelerator.device,
        )
        ema_helper = None
        if teacher_ema is not None:
            ema_decay = float(rcm_config.extra_args.get("rcm_ema_decay", 0.9999))
            ema_helper = RCMEMAHelper.from_model(student, decay=ema_decay)
            try:
                teacher_ema.load_state_dict(ema_helper.shadow, strict=False)
            except Exception as exc:
                logger.warning("Failed to prime EMA teacher: %s", exc)
        opt_bundle = build_rcm_optimizer(
            args,
            rcm_config,
            student,
            accelerator=accelerator,
            raw_config=raw_config,
            config_path=config_path,
            fake_score=fake_score,
        )

        tokenizer = None
        text_tokenizer = rcm_config.extra_args.get("text_tokenizer")
        if text_tokenizer:
            tokenizer_assets = TokenizerAssets(
                text_model=text_tokenizer,
                clean_mode=rcm_config.extra_args.get("tokenizer_clean"),
                max_length=(
                    int(rcm_config.extra_args["tokenizer_max_length"])
                    if rcm_config.extra_args.get("tokenizer_max_length") is not None
                    else None
                ),
                cache_dir=rcm_config.extra_args.get("tokenizer_cache_dir"),
            )
            tokenizer = RCMTokenizer(tokenizer_assets)

        run_distillation(
            accelerator=accelerator,
            dataset=dataset,
            teacher=teacher,
            student=student,
            teacher_ema=teacher_ema,
            fake_score=fake_score,
            optimizer=opt_bundle.optimizer,  # type: ignore[attr-defined]
            scheduler=getattr(opt_bundle, "scheduler", None),
            config=rcm_config,
            overrides=rcm_config.extra_args,
            checkpoint_callback=_build_checkpoint_callback(
                args, raw_config, rcm_config
            ),
            optimizer_train_fn=getattr(opt_bundle, "optimizer_train_fn", None),
            optimizer_eval_fn=getattr(opt_bundle, "optimizer_eval_fn", None),
            args=args,
            tokenizer=tokenizer,
            ema_helper=ema_helper,
            fake_score_optimizer=getattr(opt_bundle, "fake_score_optimizer", None),
            fake_score_scheduler=getattr(opt_bundle, "fake_score_scheduler", None),
            fake_score_optimizer_train_fn=getattr(
                opt_bundle, "fake_score_optimizer_train_fn", None
            ),
            fake_score_optimizer_eval_fn=getattr(
                opt_bundle, "fake_score_optimizer_eval_fn", None
            ),
        )
    except NotImplementedError as exc:
        logger.error("RCM pipeline is not fully implemented: %s", exc)
        raise
    except Exception:
        logger.exception("RCM pipeline execution failed.")
        return False
    finally:
        _remove_enhancement_hooks(hook_handles)

    logger.info("RCM pipeline completed successfully.")
    return True


def _create_accelerator(args: argparse.Namespace, config: RCMConfig) -> Accelerator:
    """Instantiate an ``Accelerator`` using basic heuristics."""

    cpu_flag = bool(config.cpu_debug or config.accelerator_mode == "cpu")
    kwargs: Dict[str, Any] = {}
    if cpu_flag:
        kwargs["cpu"] = True
        logger.info("RCM pipeline running in CPU debug mode.")

    accelerator = Accelerator(**kwargs)
    return accelerator


def _setup_enhancement_hooks(
    args: argparse.Namespace, accelerator: Accelerator
) -> Iterable[Any]:
    """Placeholder for enhancement hook registration (currently no-op)."""

    # TODO: integrate enhancement setup once a reusable registry exists.
    return ()


def _remove_enhancement_hooks(handles: Iterable[Any]) -> None:
    """Cleanup counterpart for :func:`_setup_enhancement_hooks`."""

    for handle in handles:
        try:
            handle.remove()  # type: ignore[attr-defined]
        except AttributeError:
            continue


def _build_checkpoint_callback(
    args: argparse.Namespace,
    raw_config: Dict[str, Any],
    rcm_config: RCMConfig,
) -> Optional[Any]:
    """Create checkpoint callback backed by Takenoko's CheckpointManager."""

    base_output = Path(getattr(args, "output_dir", "") or raw_config.get("output_dir", "output"))
    output_root = base_output / "rcm"
    output_root.mkdir(parents=True, exist_ok=True)

    args_for_checkpoint = copy.copy(args)
    args_for_checkpoint.output_dir = str(output_root)

    checkpoint_manager = CheckpointManager()
    metadata = {
        "rcm_teacher": getattr(args, "dit", ""),
        "rcm_variant": rcm_config.trainer_variant,
        "rcm_target": "wan2.2",
    }
    minimum_metadata: Dict[str, str] = {}

    dit_dtype = getattr(args, "dit_dtype", torch.float16)
    save_model_fn = checkpoint_manager.create_save_model_function(
        args=args_for_checkpoint,
        metadata=metadata,
        minimum_metadata=minimum_metadata,
        dit_dtype=dit_dtype,
    )
    remove_model_fn = checkpoint_manager.create_remove_model_function(args_for_checkpoint)

    keep_last = int(rcm_config.extra_args.get("checkpoint_keep_last", 3))
    saved_history: Deque[str] = deque()

    def _callback(state: Dict[str, Any]) -> None:
        model = state.get("model")
        if model is None:
            logger.debug("Checkpoint callback received no model reference; skipping.")
            return

        step = int(state.get("step", 0))
        ckpt_name = f"rcm_step_{step:06d}.safetensors"

        save_model_fn(
            ckpt_name,
            model,
            step,
            state.get("epoch", 0),
            force_sync_upload=False,
        )
        logger.info("RCM checkpoint written to %s", output_root / ckpt_name)

        if keep_last > 0:
            saved_history.append(ckpt_name)
            while len(saved_history) > keep_last:
                old_ckpt = saved_history.popleft()
                remove_model_fn(old_ckpt)

    return _callback
