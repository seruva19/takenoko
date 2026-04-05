import argparse
import hashlib
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm import tqdm

from common.logger import get_logger
from modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from scheduling.timestep_utils import get_noisy_model_input_and_timesteps
from utils.train_utils import compute_loss_weighting_for_sd3

logger = get_logger(__name__)


def _path_mtime(path: Optional[str]) -> Optional[float]:
    if not path or not os.path.exists(path):
        return None
    return float(os.path.getmtime(path))


def _signature_hash(signature: Dict[str, Any]) -> str:
    payload = json.dumps(signature, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _safe_abs_path(path: Optional[str]) -> Optional[str]:
    return os.path.abspath(path) if path else None


def _extract_transformer_block_index(name: str) -> Optional[int]:
    match = re.search(r"(?:^|\.)(?:model\.)?blocks\.(\d+)\.", name)
    if match is None:
        return None
    return int(match.group(1))


def _is_attention_geometry_param(name: str) -> bool:
    geometry_tokens = (
        ".self_attn.q.",
        ".self_attn.k.",
        ".self_attn.norm_q.",
        ".self_attn.norm_k.",
        ".cross_attn.q.",
        ".cross_attn.k.",
        ".cross_attn.norm_q.",
        ".cross_attn.norm_k.",
    )
    return any(token in name for token in geometry_tokens)


def _is_ewc_target_param(name: str, target: str) -> bool:
    if target == "all_trainable":
        return True
    if target == "attn_norm_bias":
        return (
            re.search(
                r"(?:^|\.)(?:self_attn|cross_attn)\.(?:norm_q|norm_k)\.weight$",
                name,
            )
            is not None
            or re.search(r"(?:^|\.)(?:self_attn|cross_attn)\.(?:q|k)\.bias$", name)
            is not None
        )
    if target == "attn_geometry":
        return _is_attention_geometry_param(name)
    return False


def _ewc_param_priority(name: str) -> tuple[int, int, str]:
    block_idx = _extract_transformer_block_index(name)
    block_sort = int(block_idx) if block_idx is not None else 10**9
    if re.search(
        r"(?:^|\.)(?:self_attn|cross_attn)\.(?:q|k|norm_q|norm_k)\.",
        name,
    ):
        tier = 0
    elif _is_attention_geometry_param(name):
        tier = 1
    elif re.search(r"(?:^|\.)(?:self_attn|cross_attn)\.(?:v|o)\.", name):
        tier = 2
    elif ".ffn." in name or ".ff." in name:
        tier = 3
    elif ".norm" in name:
        tier = 4
    elif block_idx is not None:
        tier = 5
    else:
        tier = 6
    return tier, block_sort, name


def _cap_selected_params(
    selected: List[tuple[str, torch.nn.Parameter]],
    *,
    target: str,
    max_tensors: int,
) -> List[tuple[str, torch.nn.Parameter]]:
    if max_tensors <= 0 or len(selected) <= max_tensors:
        return selected

    ranked = sorted(selected, key=lambda item: _ewc_param_priority(item[0]))
    if target != "all_trainable":
        return ranked[:max_tensors]

    by_block: Dict[int, List[tuple[str, torch.nn.Parameter]]] = {}
    non_block: List[tuple[str, torch.nn.Parameter]] = []
    for item in ranked:
        block_idx = _extract_transformer_block_index(item[0])
        if block_idx is None:
            non_block.append(item)
        else:
            by_block.setdefault(int(block_idx), []).append(item)

    picked: List[tuple[str, torch.nn.Parameter]] = []
    active_blocks = sorted(by_block.keys())
    while active_blocks and len(picked) < max_tensors:
        next_active: List[int] = []
        for block_idx in active_blocks:
            bucket = by_block[block_idx]
            if bucket:
                picked.append(bucket.pop(0))
                if len(picked) >= max_tensors:
                    break
            if bucket:
                next_active.append(block_idx)
        active_blocks = next_active

    if len(picked) < max_tensors and non_block:
        remaining = max_tensors - len(picked)
        picked.extend(non_block[:remaining])
    return picked[:max_tensors]


def _build_ewc_cache_signature(args: argparse.Namespace) -> Dict[str, Any]:
    checkpoint_path = _safe_abs_path(getattr(args, "dit", None))
    dataset_config = _safe_abs_path(getattr(args, "dataset_config", None))
    return {
        "schema": 1,
        "kind": "ewc_cache",
        "checkpoint": checkpoint_path,
        "checkpoint_mtime": _path_mtime(checkpoint_path),
        "dataset_config": dataset_config,
        "dataset_config_mtime": _path_mtime(dataset_config),
        "ewc_target": getattr(args, "ewc_target", "attn_norm_bias"),
        "ewc_num_batches": int(getattr(args, "ewc_num_batches", 8) or 0),
        "ewc_max_param_tensors": int(getattr(args, "ewc_max_param_tensors", 256) or 0),
        "freeze_early_blocks": int(getattr(args, "freeze_early_blocks", 0) or 0),
        "block_lr_scales": getattr(args, "block_lr_scales", None),
        "freeze_attn_geometry": bool(getattr(args, "freeze_attn_geometry", False)),
    }


class EwcHelper:
    def __init__(self, training_core: Any, args: argparse.Namespace):
        self.training_core = training_core
        self.args = args
        self.state: Optional[Dict[str, Any]] = None
        self.last_penalty_raw: Optional[torch.Tensor] = None
        self.last_used_tensors: Optional[torch.Tensor] = None
        self.last_skipped_tensors: Optional[torch.Tensor] = None

    def build_or_load(
        self,
        accelerator: Accelerator,
        transformer: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        optimizer: Any,
        network_dtype: torch.dtype,
        *,
        fused_step_state: Optional[Dict[str, bool]] = None,
        timestep_distribution: Any = None,
    ) -> Optional[Dict[str, Any]]:
        if float(getattr(self.args, "ewc_lambda", 0.0) or 0.0) <= 0.0:
            return None

        cache_path = getattr(self.args, "ewc_cache_path", None)
        cache_rebuild = bool(getattr(self.args, "ewc_cache_rebuild", False))
        signature = _build_ewc_cache_signature(self.args)
        loaded = False
        if cache_path and not cache_rebuild:
            payload = self._load_cache(cache_path, signature, transformer, accelerator.device)
            if payload is not None:
                self.state = payload
                loaded = True
        elif cache_path and cache_rebuild:
            logger.info("EWC cache rebuild requested; ignoring existing cache: %s", cache_path)

        if self.state is None:
            if fused_step_state is not None:
                fused_step_state["suspend_step"] = True
            try:
                self.state = self._build_fisher_stats(
                    accelerator,
                    transformer,
                    train_dataloader,
                    optimizer,
                    network_dtype,
                    timestep_distribution=timestep_distribution,
                )
            finally:
                if fused_step_state is not None:
                    fused_step_state["suspend_step"] = False
                    optimizer.zero_grad(set_to_none=True)
            if self.state is not None and cache_path and not loaded:
                self._save_cache(cache_path, signature, self.state)
        return self.state

    def _build_fisher_stats(
        self,
        accelerator: Accelerator,
        transformer: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        optimizer: Any,
        network_dtype: torch.dtype,
        *,
        timestep_distribution: Any = None,
    ) -> Optional[Dict[str, Any]]:
        ewc_num_batches = int(getattr(self.args, "ewc_num_batches", 8) or 0)
        if ewc_num_batches <= 0:
            logger.warning("ewc_lambda > 0 but ewc_num_batches <= 0. Skipping EWC.")
            return None

        ewc_target = str(getattr(self.args, "ewc_target", "attn_norm_bias") or "attn_norm_bias")
        ewc_max_param_tensors = int(getattr(self.args, "ewc_max_param_tensors", 256) or 0)
        selected = [
            (name, param)
            for name, param in transformer.named_parameters()
            if param.requires_grad and _is_ewc_target_param(name, ewc_target)
        ]
        selected = _cap_selected_params(
            selected,
            target=ewc_target,
            max_tensors=ewc_max_param_tensors,
        )
        if not selected:
            logger.warning(
                "EWC requested but no matching trainable parameters were selected (target=%s).",
                ewc_target,
            )
            return None

        logger.info(
            "Building Fisher/EWC stats on %d parameter tensors (target=%s, batches=%d).",
            len(selected),
            ewc_target,
            ewc_num_batches,
        )
        start_time = time.time()
        theta_ref = {name: param.detach().cpu().float().clone() for name, param in selected}
        fisher_acc = {name: torch.zeros_like(theta_ref[name]) for name, _ in selected}
        valid_batches = 0
        skipped_batches = 0
        was_training = transformer.training
        noise_scheduler = FlowMatchDiscreteScheduler(
            shift=getattr(self.args, "discrete_flow_shift", 3.0),
            reverse=True,
            solver="euler",
        )
        pbar = tqdm(
            total=ewc_num_batches,
            desc="prep: fisher/ewc",
            leave=False,
            disable=not accelerator.is_local_main_process,
        )
        transformer.eval()
        try:
            for batch in train_dataloader:
                if valid_batches >= ewc_num_batches:
                    break
                latents = batch.get("latents")
                if not isinstance(latents, torch.Tensor) or latents.dim() != 5:
                    skipped_batches += 1
                    continue

                latents = latents.to(device=accelerator.device, dtype=network_dtype)
                noise = torch.randn_like(latents)
                noisy_model_input, timesteps, _ = get_noisy_model_input_and_timesteps(
                    self.args,
                    noise,
                    latents,
                    noise_scheduler,
                    accelerator.device,
                    network_dtype,
                    timestep_distribution=timestep_distribution,
                    item_info=batch.get("item_info"),
                )
                weighting = compute_loss_weighting_for_sd3(
                    getattr(self.args, "weighting_scheme", "none"),
                    noise_scheduler,
                    timesteps,
                    accelerator.device,
                    network_dtype,
                )
                model_result = self.training_core.call_dit(
                    self.args,
                    accelerator,
                    transformer,
                    latents,
                    batch,
                    noise,
                    noisy_model_input,
                    timesteps,
                    network_dtype,
                    model_timesteps_override=timesteps,
                    apply_stable_velocity_target=False,
                )
                if not isinstance(model_result, tuple) or len(model_result) < 2:
                    skipped_batches += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue
                model_pred = model_result[0]
                target = model_result[1]
                pred = model_pred.to(device=target.device, dtype=network_dtype)
                loss_type = getattr(self.args, "loss_type", "mse")
                huber_delta = float(getattr(self.args, "huber_delta", 1.0))
                if loss_type in ("mae", "l1"):
                    loss = F.l1_loss(pred, target, reduction="none")
                elif loss_type in ("huber", "smooth_l1"):
                    loss = F.smooth_l1_loss(
                        pred,
                        target,
                        reduction="none",
                        beta=huber_delta,
                    )
                else:
                    loss = F.mse_loss(pred, target, reduction="none")
                if weighting is not None:
                    weight_tensor = weighting
                    while (
                        isinstance(weight_tensor, torch.Tensor)
                        and weight_tensor.dim() > loss.dim()
                        and weight_tensor.shape[-1] == 1
                    ):
                        weight_tensor = weight_tensor.squeeze(-1)
                    loss = loss * weight_tensor
                loss = loss.mean()

                accelerator.backward(loss)
                for name, param in selected:
                    if param.grad is not None:
                        fisher_acc[name].add_(param.grad.detach().float().pow(2).cpu())
                optimizer.zero_grad(set_to_none=True)
                valid_batches += 1
                pbar.update(1)
        finally:
            pbar.close()
            if was_training:
                transformer.train()
            optimizer.zero_grad(set_to_none=True)

        if valid_batches == 0:
            logger.warning("EWC statistics build produced 0 valid batches; skipping EWC.")
            return None
        for name in list(fisher_acc.keys()):
            fisher_acc[name] = fisher_acc[name].div(float(valid_batches)).clamp_min(1e-12)
        logger.info(
            "EWC stats built with %d valid batches (%d skipped) in %.1fs.",
            valid_batches,
            skipped_batches,
            time.time() - start_time,
        )
        return {
            "params": [(name, param) for name, param in selected],
            "theta_ref": theta_ref,
            "fisher": fisher_acc,
        }

    def compute_penalty(
        self,
        *,
        dtype: torch.dtype,
        target_device: Optional[torch.device] = None,
    ) -> tuple[Optional[torch.Tensor], int, int]:
        self.last_penalty_raw = None
        self.last_used_tensors = None
        self.last_skipped_tensors = None
        if not self.state:
            return None, 0, 0
        compute_dtype = dtype if dtype in (torch.float16, torch.bfloat16) else torch.float32
        penalty: Optional[torch.Tensor] = None
        used_params = 0
        skipped_params = 0
        for name, param in self.state["params"]:
            if target_device is not None and param.device != target_device:
                skipped_params += 1
                continue
            theta = self.state["theta_ref"][name].to(
                device=param.device,
                dtype=compute_dtype,
                non_blocking=True,
            )
            fisher = self.state["fisher"][name].to(
                device=param.device,
                dtype=compute_dtype,
                non_blocking=True,
            )
            diff = param.to(compute_dtype) - theta
            term = (fisher * diff.square()).mean()
            penalty = term if penalty is None else (penalty + term)
            used_params += 1

        if penalty is None:
            return None, used_params, skipped_params
        if target_device is not None:
            penalty = penalty.to(device=target_device, dtype=compute_dtype)
        self.last_penalty_raw = penalty.detach()
        self.last_used_tensors = penalty.detach().new_tensor(float(used_params))
        self.last_skipped_tensors = penalty.detach().new_tensor(float(skipped_params))
        return penalty, used_params, skipped_params

    def _load_cache(
        self,
        path: str,
        signature: Dict[str, Any],
        transformer: torch.nn.Module,
        target_device: torch.device,
    ) -> Optional[Dict[str, Any]]:
        if not path or not os.path.exists(path):
            return None
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception:
            logger.exception("Failed to load EWC cache, rebuilding: %s", path)
            return None
        if not isinstance(payload, dict) or payload.get("signature_hash") != _signature_hash(signature):
            logger.info("EWC cache signature mismatch; rebuilding: %s", path)
            return None

        theta_ref = payload.get("theta_ref")
        fisher = payload.get("fisher")
        if not isinstance(theta_ref, dict) or not isinstance(fisher, dict):
            return None

        params: List[tuple[str, torch.nn.Parameter]] = []
        named_parameters = dict(transformer.named_parameters())
        for name in theta_ref.keys():
            param = named_parameters.get(name)
            if param is None or not param.requires_grad:
                continue
            if param.device != target_device:
                continue
            params.append((name, param))
        if not params:
            return None
        logger.info("Loaded EWC cache: %s (params=%d)", path, len(params))
        return {"params": params, "theta_ref": theta_ref, "fisher": fisher}

    def _save_cache(
        self,
        path: str,
        signature: Dict[str, Any],
        state: Dict[str, Any],
    ) -> None:
        payload = {
            "version": 1,
            "signature_hash": _signature_hash(signature),
            "signature": signature,
            "created_at": float(time.time()),
            "theta_ref": state.get("theta_ref", {}),
            "fisher": state.get("fisher", {}),
        }
        cache_dir = os.path.dirname(path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        torch.save(payload, path)
        logger.info("Saved EWC cache: %s", path)
