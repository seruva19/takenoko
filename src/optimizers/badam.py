"""Takenoko-native BAdam optimizer wrapper.

BAdam applies block-coordinate training on top of a normal optimizer. Only the
current block and configured always-active parameters require gradients; inactive
optimizer state can be purged at each switch to keep memory bounded.
"""

from __future__ import annotations

import random
import re
import warnings
from collections import defaultdict
from typing import Any, Iterable, Sequence

import torch


BADAM_OPTIMIZER_ALIASES = {"badam", "blockadam", "block_optimizer", "blockoptimizer"}
BADAM_RATIO_OPTIMIZER_ALIASES = {"badamratio", "badam_ratio", "blockadamratio"}


def is_badam_optimizer_type(optimizer_type: str) -> bool:
    return optimizer_type.lower() in BADAM_OPTIMIZER_ALIASES


def is_badam_ratio_optimizer_type(optimizer_type: str) -> bool:
    return optimizer_type.lower() in BADAM_RATIO_OPTIMIZER_ALIASES


def flatten_optimizer_params(params_or_groups: Sequence[Any]) -> list[torch.nn.Parameter]:
    """Flatten torch optimizer params/groups while preserving first occurrence order."""

    params: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for item in params_or_groups:
        if isinstance(item, dict):
            group_params = item.get("params", [])
        else:
            group_params = [item]
        for param in group_params:
            if not isinstance(param, torch.nn.Parameter):
                continue
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            params.append(param)
    return params


def normalize_block_prefixes(
    prefixes: Iterable[str | Iterable[str]],
) -> list[tuple[str, ...]]:
    normalized: list[tuple[str, ...]] = []
    for item in prefixes:
        if isinstance(item, str):
            group = (item,)
        else:
            group = tuple(prefix for prefix in item if isinstance(prefix, str))
        if not group:
            continue
        normalized.append(group)
    return normalized


def infer_wan_block_prefixes(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
) -> list[tuple[str, ...]]:
    """Infer Wan transformer block prefixes from trainable parameter names."""

    prefixes_by_index: dict[int, str] = {}
    for name, _param in named_parameters:
        match = re.match(r"^(.*?blocks\.(\d+)\.)", name)
        if match is None:
            continue
        prefixes_by_index[int(match.group(2))] = match.group(1)

    return [(prefixes_by_index[index],) for index in sorted(prefixes_by_index)]


def infer_transformer_block_prefixes(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    *,
    include_embedding: bool = False,
    include_lm_head: bool = False,
) -> list[tuple[str, ...]]:
    """Infer upstream BAdam-style prefixes, with Wan `blocks.N.` support."""

    wan_blocks = infer_wan_block_prefixes(named_parameters)
    if wan_blocks:
        prefixes = list(wan_blocks)
        block_flat = [prefix for group in prefixes for prefix in group]
        embed_prefixes: dict[str, str] = {}
        remainder: list[str] = []
        embed_pattern = re.compile(r"^(.*?embed[^.]*\.)")
        for name, _param in named_parameters:
            if _matches_any(name, block_flat):
                continue
            embed_match = embed_pattern.match(name)
            if embed_match is not None and include_embedding:
                embed_prefixes[embed_match.group(1)] = embed_match.group(1)
            else:
                remainder.append(name)
        if embed_prefixes:
            prefixes = [(prefix,) for prefix in embed_prefixes.values()] + prefixes
        if include_lm_head and remainder:
            prefixes.append(tuple(remainder))
    else:
        prefixes_by_index: dict[int, str] = {}
        remainder: list[str] = []
        embed_prefixes: dict[str, str] = {}
        layer_pattern = re.compile(r"^(.*?layers\.(\d+)\.)")
        embed_pattern = re.compile(r"^(.*?embed[^.]*\.)")
        for name, _param in named_parameters:
            layer_match = layer_pattern.match(name)
            if layer_match is not None:
                prefixes_by_index[int(layer_match.group(2))] = layer_match.group(1)
                continue
            embed_match = embed_pattern.match(name)
            if embed_match is not None and include_embedding:
                embed_prefixes[embed_match.group(1)] = embed_match.group(1)
                continue
            remainder.append(name)
        prefixes = [(prefix,) for prefix in embed_prefixes.values()]
        prefixes.extend(
            (prefixes_by_index[index],) for index in sorted(prefixes_by_index)
        )
        if include_lm_head and remainder:
            prefixes.append(tuple(remainder))

    return prefixes


def _matches_any(name: str, prefixes: Iterable[str]) -> bool:
    for prefix in prefixes:
        if prefix.startswith("re:"):
            if re.search(prefix[3:], name):
                return True
        elif prefix in name:
            return True
    return False


class TakenokoBlockOptimizer(torch.optim.Optimizer):
    """Block-coordinate wrapper for a base optimizer.

    The wrapper keeps the base optimizer's parameter groups stable so schedulers
    and Accelerate wrappers continue to see the same groups throughout training.
    """

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
        block_prefixes: Sequence[tuple[str, ...]],
        *,
        switch_block_every: int = 100,
        switch_mode: str = "random",
        start_block: int | None = None,
        always_active_prefixes: Sequence[str] | None = None,
        include_non_block: bool = True,
        use_fp32_active_copy: bool = True,
        purge_inactive_state: bool = True,
        reset_state_on_switch: bool = True,
        verbose: int = 1,
        logger: Any | None = None,
    ) -> None:
        if switch_block_every < 1:
            raise ValueError("switch_block_every must be >= 1")
        if switch_mode not in {"random", "ascending", "descending", "fixed"}:
            raise ValueError("switch_mode must be random, ascending, descending, or fixed")
        if not block_prefixes:
            raise ValueError("BAdam requires at least one block prefix")

        self.base_optimizer = base_optimizer
        self.named_parameters_list = list(named_parameters)
        self.block_prefixes = list(block_prefixes)
        self.switch_block_every = int(switch_block_every)
        self.switch_mode = switch_mode
        self.always_active_prefixes = tuple(always_active_prefixes or ())
        self.include_non_block = bool(include_non_block)
        self.use_fp32_active_copy = bool(use_fp32_active_copy)
        self.purge_inactive_state = bool(purge_inactive_state)
        self.reset_state_on_switch = bool(reset_state_on_switch)
        self.verbose = int(verbose)
        self.logger = logger
        self.global_step = 0
        self._random_order: list[int] = []
        self._active_param_ids: set[int] = set()
        self._lp_to_hp: dict[torch.nn.Parameter, torch.nn.Parameter] = {}
        self._hp_to_lp: dict[torch.nn.Parameter, torch.nn.Parameter] = {}
        self._active_source_group_indices: list[int] = []
        self._block_managed_param_ids = self._collect_block_managed_param_ids()
        # Gradient-release (fused-like) state — populated lazily via enable_gradient_release()
        # so it can be wired up after accelerator.prepare(). Until enabled, classical step() runs.
        self._gradient_release_enabled: bool = False
        self._gr_accelerator: Any = None
        self._gr_max_grad_norm: float = 0.0
        self._gr_fused_step_state: dict[str, Any] | None = None
        self._gr_hook_handles: list[Any] = []

        fp32_params = [name for name, param in self.named_parameters_list if param.dtype == torch.float32]
        if fp32_params and self.use_fp32_active_copy:
            warnings.warn(
                "BAdam expects the model to be loaded in fp16/bf16 for memory savings, "
                f"but found fp32 trainable parameters: {fp32_params}",
                RuntimeWarning,
                stacklevel=2,
            )

        if start_block is not None:
            if start_block >= len(self.block_prefixes):
                raise ValueError(
                    f"start_block={start_block} is outside {len(self.block_prefixes)} BAdam blocks"
                )
            self.current_block_idx = int(start_block)
        elif self.switch_mode == "descending":
            self.current_block_idx = len(self.block_prefixes) - 1
        elif self.switch_mode == "random":
            self.current_block_idx = self._pop_random_block()
        else:
            self.current_block_idx = 0

        super().__init__(base_optimizer.param_groups, base_optimizer.defaults)
        self.param_groups = base_optimizer.param_groups
        self.defaults = base_optimizer.defaults
        self.state = base_optimizer.state

        self.switch_trainable_params(advance=False)

    def __getstate__(self) -> dict[str, Any]:
        return {
            "base_optimizer": self.base_optimizer,
            "global_step": self.global_step,
            "current_block_idx": self.current_block_idx,
            "random_order": self._random_order,
        }

    def _log(self, level: str, message: str, *args: Any) -> None:
        if self.logger is None:
            return
        log_fn = getattr(self.logger, level, None)
        if callable(log_fn):
            log_fn(message, *args)

    def _collect_block_managed_param_ids(self) -> set[int]:
        managed: set[int] = set()
        flat_prefixes = [prefix for group in self.block_prefixes for prefix in group]
        for name, param in self.named_parameters_list:
            if _matches_any(name, flat_prefixes):
                managed.add(id(param))
        return managed

    def _pop_random_block(self) -> int:
        if not self._random_order:
            self._random_order = list(range(len(self.block_prefixes)))
            random.shuffle(self._random_order)
            if self.verbose >= 2:
                self._log("info", "BAdam next random block order: %s", self._random_order)
        return self._random_order.pop()

    def _advance_block_idx(self) -> None:
        if self.switch_mode == "random":
            self.current_block_idx = self._pop_random_block()
        elif self.switch_mode == "ascending":
            self.current_block_idx = (self.current_block_idx + 1) % len(self.block_prefixes)
        elif self.switch_mode == "descending":
            self.current_block_idx = (self.current_block_idx - 1) % len(self.block_prefixes)
        elif self.switch_mode == "fixed":
            return

    def _is_always_active(self, name: str, param: torch.nn.Parameter) -> bool:
        if _matches_any(name, self.always_active_prefixes):
            return True
        return self.include_non_block and id(param) not in self._block_managed_param_ids

    def switch_trainable_params(self, *, advance: bool = True) -> None:
        if advance:
            self._advance_block_idx()

        active_prefixes = self.block_prefixes[self.current_block_idx]
        active_ids: set[int] = set()
        active_names: list[str] = []
        inactive_count = 0

        for name, param in self.named_parameters_list:
            is_active = self._is_always_active(name, param) or _matches_any(
                name,
                active_prefixes,
            )
            param.requires_grad_(is_active)
            if is_active:
                active_ids.add(id(param))
                if self.verbose >= 2:
                    active_names.append(name)
            else:
                inactive_count += 1
                param.grad = None
                if self.purge_inactive_state:
                    self.base_optimizer.state.pop(param, None)

        self._active_param_ids = active_ids

        if self.use_fp32_active_copy:
            self._rebuild_active_fp32_param_groups()
        else:
            self.base_optimizer.param_groups = self.param_groups
            if self.reset_state_on_switch:
                self.base_optimizer.state.clear()

        # Re-arm gradient-release hooks if enabled — old hooks pointed at the
        # previous active block's HP copies, which were just discarded.
        if self._gradient_release_enabled:
            self._refresh_gradient_release_hooks()

        if self.verbose >= 1:
            self._log(
                "info",
                "BAdam active block %d/%d prefixes=%s active_params=%d inactive_params=%d",
                self.current_block_idx,
                len(self.block_prefixes),
                list(active_prefixes),
                len(active_ids),
                inactive_count,
            )
        if active_names:
            self._log("info", "BAdam active parameter names: %s", active_names)

    def _clone_group_options(self, group: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in group.items() if key != "params"}

    def _rebuild_active_fp32_param_groups(self) -> None:
        self._lp_to_hp = {}
        self._hp_to_lp = {}
        self._active_source_group_indices = []
        active_groups: list[dict[str, Any]] = []

        for source_group_idx, group in enumerate(self.param_groups):
            active_hp_params: list[torch.nn.Parameter] = []
            for param in group.get("params", []):
                if id(param) not in self._active_param_ids:
                    continue
                hp_param = torch.nn.Parameter(
                    param.detach().clone().float(),
                    requires_grad=True,
                )
                self._lp_to_hp[param] = hp_param
                self._hp_to_lp[hp_param] = param
                active_hp_params.append(hp_param)

            if not active_hp_params:
                continue
            active_group = self._clone_group_options(group)
            active_group["params"] = active_hp_params
            active_group["_badam_source_group_idx"] = source_group_idx
            active_groups.append(active_group)
            self._active_source_group_indices.append(source_group_idx)

        if not active_groups:
            raise RuntimeError("BAdam active block has no optimizer parameters.")
        self.base_optimizer.param_groups = active_groups
        if self.reset_state_on_switch:
            self.base_optimizer.state.clear()

    def _sync_active_optimizer_lrs(self) -> None:
        if not self.use_fp32_active_copy:
            return
        for active_group in self.base_optimizer.param_groups:
            source_group_idx = active_group.get("_badam_source_group_idx")
            if source_group_idx is None:
                continue
            source_group = self.param_groups[int(source_group_idx)]
            for key in ("lr", "weight_decay"):
                if key in source_group:
                    active_group[key] = source_group[key]

    def _move_grads_to_hp(self) -> None:
        for lp_param, hp_param in self._lp_to_hp.items():
            if lp_param.grad is None:
                hp_param.grad = None
                continue
            hp_param.grad = lp_param.grad.detach().float()
            lp_param.grad = None

    def _copy_hp_to_lp(self) -> None:
        for hp_param, lp_param in self._hp_to_lp.items():
            lp_param.data.copy_(hp_param.detach().to(dtype=lp_param.dtype))

    def synchronize_for_checkpoint(self) -> None:
        if self.use_fp32_active_copy:
            self._copy_hp_to_lp()

    # --- gradient-release (fused-like) ---------------------------------------

    def enable_gradient_release(
        self,
        *,
        accelerator: Any = None,
        max_grad_norm: float = 0.0,
        fused_step_state: dict[str, Any] | None = None,
    ) -> None:
        """Enable per-parameter step via post-accumulate-grad hooks.

        Hooks release LP grad immediately after accumulate, run a single-param
        AdamW (or other base) step on the HP fp32 copy, then write the updated
        weight back to LP and free HP grad. Net effect: gradient peak shrinks
        from a full active-block worth (~1.4 GB) to per-tensor.

        Must be called AFTER accelerator.prepare() so hooks attach to the
        underlying parameters that backward will populate.
        """
        if not self.use_fp32_active_copy:
            raise RuntimeError(
                "BAdam gradient-release currently requires use_fp32_active_copy=True; "
                "the per-param step path operates on HP fp32 copies."
            )
        self._gradient_release_enabled = True
        self._gr_accelerator = accelerator
        self._gr_max_grad_norm = float(max_grad_norm or 0.0)
        self._gr_fused_step_state = fused_step_state
        # Re-arm hooks for currently active block.
        self._refresh_gradient_release_hooks()
        if self.verbose >= 1:
            self._log(
                "info",
                "BAdam gradient-release enabled (max_grad_norm=%.4f, defer_aware=%s)",
                self._gr_max_grad_norm,
                fused_step_state is not None,
            )

    def _unregister_gradient_release_hooks(self) -> None:
        for handle in self._gr_hook_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._gr_hook_handles = []

    def _refresh_gradient_release_hooks(self) -> None:
        if not self._gradient_release_enabled:
            return
        self._unregister_gradient_release_hooks()
        for lp_param, hp_param in self._lp_to_hp.items():
            handle = lp_param.register_post_accumulate_grad_hook(
                self._make_grad_release_hook(lp_param, hp_param)
            )
            self._gr_hook_handles.append(handle)

    def _make_grad_release_hook(
        self,
        lp_param: torch.nn.Parameter,
        hp_param: torch.nn.Parameter,
    ):
        def hook(p: torch.Tensor) -> None:
            if p.grad is None:
                return
            accel = self._gr_accelerator
            sync = True if accel is None else bool(getattr(accel, "sync_gradients", True))
            # Per-param grad clipping (matches Adafactor fused trainer path).
            if sync and self._gr_max_grad_norm > 0 and accel is not None:
                accel.clip_grad_norm_(p, self._gr_max_grad_norm)
            # Move LP grad → HP grad and free LP grad immediately for VRAM.
            hp_param.grad = p.grad.detach().float()
            p.grad = None
            if not sync:
                # Gradient accumulation in progress; let HP grad accumulate.
                return
            fss = self._gr_fused_step_state
            if fss is not None and (
                bool(fss.get("defer_step")) or bool(fss.get("suspend_step"))
            ):
                # Motion preservation deferred path: hold HP grad until step() runs
                # the classical batch update.
                return
            # Apply per-param step on HP, write back, free HP grad.
            self._step_single_hp_param(hp_param)
            lp_param.data.copy_(hp_param.detach().to(lp_param.dtype))
            hp_param.grad = None

        return hook

    def _step_single_hp_param(self, hp_param: torch.nn.Parameter) -> None:
        """Run base optimizer for exactly one HP parameter via temporary group swap."""
        if hp_param.grad is None:
            return
        saved_groups = self.base_optimizer.param_groups
        single_group: dict[str, Any] | None = None
        for group in saved_groups:
            if any(p is hp_param for p in group.get("params", [])):
                single_group = {**group, "params": [hp_param]}
                break
        if single_group is None:
            return
        self.base_optimizer.param_groups = [single_group]
        try:
            self.base_optimizer.step()
        finally:
            self.base_optimizer.param_groups = saved_groups

    def _flush_deferred_grads(self) -> None:
        """Apply optimizer step to any HP params that still hold grads.

        Used at step() time when a deferred batch update arrives (motion
        preservation defer path or accumulation boundary).
        """
        any_pending = any(hp.grad is not None for hp in self._hp_to_lp)
        if not any_pending:
            return
        self.base_optimizer.step()
        for hp_param, lp_param in self._hp_to_lp.items():
            if hp_param.grad is not None:
                lp_param.data.copy_(hp_param.detach().to(lp_param.dtype))
                hp_param.grad = None

    def _clear_inactive_grads(self) -> None:
        for name, param in self.named_parameters_list:
            if id(param) not in self._active_param_ids and not self._is_always_active(
                name,
                param,
            ):
                param.grad = None

    def step(self, closure: Any | None = None) -> Any:
        if self._gradient_release_enabled:
            # Per-param updates already happened in post-accumulate hooks during
            # backward. Only deferred grads (motion preservation defer path) and
            # the switching counter remain.
            self._sync_active_optimizer_lrs()
            self._flush_deferred_grads()
            self._clear_inactive_grads()
            self.global_step += 1
            if self.global_step % self.switch_block_every == 0:
                self.switch_trainable_params(advance=True)
            return None
        self._clear_inactive_grads()
        self._sync_active_optimizer_lrs()
        if self.use_fp32_active_copy:
            self._move_grads_to_hp()
        if closure is None:
            loss = self.base_optimizer.step()
        else:
            loss = self.base_optimizer.step(closure=closure)
        if self.use_fp32_active_copy:
            self._copy_hp_to_lp()
            for hp_param in self._hp_to_lp:
                hp_param.grad = None
        self.global_step += 1
        if self.global_step % self.switch_block_every == 0:
            self.switch_trainable_params(advance=True)
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)
        if not set_to_none:
            for name, param in self.named_parameters_list:
                if id(param) not in self._active_param_ids and not self._is_always_active(
                    name,
                    param,
                ):
                    param.grad = None

    def train(self) -> None:
        train_fn = getattr(self.base_optimizer, "train", None)
        if callable(train_fn):
            train_fn()

    def eval(self) -> None:
        eval_fn = getattr(self.base_optimizer, "eval", None)
        if callable(eval_fn):
            eval_fn()

    def state_dict(self) -> dict[str, Any]:
        state = self.base_optimizer.state_dict()
        state["_takenoko_badam"] = {
            "global_step": self.global_step,
            "current_block_idx": self.current_block_idx,
            "random_order": list(self._random_order),
            "use_fp32_active_copy": self.use_fp32_active_copy,
        }
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        state_dict = dict(state_dict)
        badam_state = state_dict.pop("_takenoko_badam", None)

        if isinstance(badam_state, dict):
            self.global_step = int(badam_state.get("global_step", self.global_step))
            current_block_idx = int(
                badam_state.get("current_block_idx", self.current_block_idx)
            )
            if 0 <= current_block_idx < len(self.block_prefixes):
                self.current_block_idx = current_block_idx
            random_order = badam_state.get("random_order", [])
            if isinstance(random_order, list) and all(
                isinstance(item, int) for item in random_order
            ):
                self._random_order = list(random_order)
        self.switch_trainable_params(advance=False)
        self.base_optimizer.load_state_dict(state_dict)
        self.state = self.base_optimizer.state


class TakenokoBlockOptimizerRatio(torch.optim.Optimizer):
    """Sparse ratio-update BAdam variant adapted from upstream BlockOptimizerRatio."""

    def __init__(
        self,
        param_groups: Sequence[Any],
        named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
        *,
        update_ratio: float = 0.1,
        switch_every: int = 100,
        preserve_threshold: int = 100,
        mask_mode: str = "adjacent",
        lr: float = 1e-5,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        keep_mask: bool = True,
        include_embedding: bool = False,
        include_lm_head: bool = False,
        verbose: int = 1,
    ) -> None:
        if not 0.0 < update_ratio <= 1.0:
            raise ValueError("update_ratio must be in (0, 1]")
        if switch_every < 1:
            raise ValueError("switch_every must be >= 1")
        if mask_mode not in {"adjacent", "scatter"}:
            raise ValueError("mask_mode must be 'adjacent' or 'scatter'")

        self.update_ratio = float(update_ratio)
        self.switch_every = int(switch_every)
        self.preserve_threshold = int(preserve_threshold)
        self.mask_mode = mask_mode
        self.keep_mask = bool(keep_mask)
        self.include_embedding = bool(include_embedding)
        self.include_lm_head = bool(include_lm_head)
        self.verbose = int(verbose)
        self.global_step = 0
        self.named_parameters_list = list(named_parameters)
        self.sparse_dict: defaultdict[torch.nn.Parameter, dict[str, Any]] = defaultdict(dict)
        self.mask_dict: dict[torch.Size, torch.Tensor | None] = {}

        optimizer_groups = self._normalize_param_groups(param_groups)
        defaults = {"lr": lr, "betas": betas, "eps": eps}
        super().__init__(optimizer_groups, defaults)

        if not self.include_embedding:
            self._set_matching_requires_grad("embed", False)
        if not self.include_lm_head:
            self._set_matching_requires_grad("lm_head", False)

        hook = self._sparse_update_hook()
        for _name, param in self.named_parameters_list:
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(hook)
            self.sparse_dict[param]["offset"] = 0
            self.sparse_dict[param]["seed"] = int(
                torch.randint(0, 1000, (1,), device=param.device).item()
            )

    @staticmethod
    def _normalize_param_groups(param_groups: Sequence[Any]) -> list[Any]:
        normalized: list[Any] = []
        for item in param_groups:
            if isinstance(item, dict):
                normalized.append(dict(item))
            else:
                normalized.append(item)
        return normalized

    def _set_matching_requires_grad(self, needle: str, value: bool) -> None:
        for name, param in self.named_parameters_list:
            if needle in name:
                param.requires_grad_(value)

    def _generate_mask_adjacent(
        self,
        param: torch.nn.Parameter,
        ratio: float,
        offset: int,
    ) -> torch.Tensor:
        num_elements = param.numel()
        num_ones = max(1, int(num_elements * ratio))
        if offset + num_ones > num_elements:
            i1 = torch.arange(
                0,
                offset + num_ones - num_elements,
                device=param.device,
            )
            i2 = torch.arange(offset, num_elements, device=param.device)
            flat_indices = torch.cat([i1, i2]).unsqueeze(0)
        else:
            flat_indices = torch.arange(
                offset,
                min(offset + num_ones, num_elements),
                device=param.device,
            ).unsqueeze(0)
        unraveled = torch.vstack(torch.unravel_index(flat_indices, param.size()))
        values = torch.ones(
            num_ones,
            device=param.device,
            dtype=param.dtype,
        )
        return torch.sparse_coo_tensor(unraveled, values, param.shape)

    def _generate_mask_scatter(
        self,
        param: torch.nn.Parameter,
        ratio: float,
        offset: int,
    ) -> torch.Tensor:
        num_elements = param.numel()
        num_ones = max(1, int(num_elements * ratio))
        generator = torch.Generator(device=param.device)
        generator.manual_seed(int(self.sparse_dict[param]["seed"]))
        randperm = torch.randperm(num_elements, device=param.device, generator=generator)
        if offset + num_ones > num_elements:
            flat_indices = torch.cat(
                [randperm[offset:], randperm[: offset + num_ones - num_elements]]
            )
        else:
            flat_indices = randperm[offset : offset + num_ones]
        unraveled = torch.vstack(torch.unravel_index(flat_indices, param.size()))
        values = torch.ones(
            num_ones,
            device=param.device,
            dtype=param.dtype,
        )
        return torch.sparse_coo_tensor(unraveled, values, param.shape)

    def _sparse_update_hook(self):
        def hook(param: torch.nn.Parameter) -> None:
            if param.grad is None:
                return
            num_elements = param.numel()
            offset = int(self.sparse_dict[param]["offset"])
            update_ratio = float(self.sparse_dict[param].get("update_ratio", self.update_ratio))

            if num_elements < self.preserve_threshold or update_ratio >= 1.0:
                param.grad = param.grad.add(1e-9).to_sparse()
                return

            mask = self.mask_dict.get(param.shape)
            if mask is None:
                if self.mask_mode == "adjacent":
                    mask = self._generate_mask_adjacent(param, update_ratio, offset)
                else:
                    mask = self._generate_mask_scatter(param, update_ratio, offset)
                if self.keep_mask:
                    self.mask_dict[param.shape] = mask

            param.grad = param.grad.sparse_mask(mask)
            if (self.global_step + 1) % self.switch_every == 0:
                next_offset = (offset + int(num_elements * update_ratio)) % num_elements
                self.sparse_dict[param]["offset"] = next_offset
                self.mask_dict[param.shape] = None

        return hook

    def _reset_state_dict(self) -> None:
        for group in self.param_groups:
            for param in group["params"]:
                self.state[param] = {}

    @torch.no_grad()
    def step(self, closure: Any | None = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            maximize = group.get("maximize", False)
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if not grad.is_sparse:
                    raise RuntimeError(
                        "BAdamRatio requires sparse gradients. Ensure the "
                        "post-accumulate gradient hook is active."
                    )
                grad = grad.coalesce()
                grad_values = grad._values()
                grad_indices = grad._indices()
                if maximize:
                    grad_values = -grad_values

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(grad_values)
                    state["exp_avg_sq"] = torch.zeros_like(grad_values)
                state["step"] += 1
                step = int(state["step"])
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                exp_avg.add_(grad_values.sub(exp_avg).mul_(1 - beta1))
                exp_avg_sq.add_(grad_values.pow(2).sub(exp_avg_sq).mul_(1 - beta2))
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                step_size = lr * (bias_correction2**0.5) / bias_correction1
                update_values = exp_avg / (exp_avg_sq.sqrt().add(eps))
                update = torch.sparse_coo_tensor(
                    grad_indices,
                    update_values,
                    param.shape,
                    device=param.device,
                    dtype=param.dtype,
                )
                param.add_(update, alpha=-step_size)

        self.global_step += 1
        if self.global_step % self.switch_every == 0:
            self._reset_state_dict()
        return loss



def create_badam_optimizer(
    args: Any,
    transformer: torch.nn.Module,
    trainable_params: Sequence[Any],
    base_optimizer: torch.optim.Optimizer,
    *,
    distributed_context: dict[str, Any] | None = None,
    logger: Any | None = None,
) -> TakenokoBlockOptimizer:
    """Wrap an already-created optimizer with Takenoko BAdam."""

    distributed_context = distributed_context or {}
    world_size = int(distributed_context.get("world_size", 1) or 1)
    if world_size > 1 and not bool(getattr(args, "badam_allow_distributed", False)):
        raise ValueError(
            "BAdam changes requires_grad between blocks and is currently supported "
            "only for single-process training in Takenoko. Set badam_allow_distributed "
            "only after validating your distributed setup."
        )

    flat_trainable = flatten_optimizer_params(list(trainable_params))
    trainable_ids = {id(param) for param in flat_trainable}
    named_parameters = [
        (name, param)
        for name, param in transformer.named_parameters()
        if id(param) in trainable_ids
    ]

    matched_ids = {id(param) for _name, param in named_parameters}
    unmatched_count = len(trainable_ids - matched_ids)
    allow_unmatched = bool(getattr(args, "badam_allow_unmatched_params", False))
    if unmatched_count and not allow_unmatched:
        raise ValueError(
            "BAdam requires trainable parameters to be named by the transformer. "
            f"Found {unmatched_count} unmatched trainable parameter(s). This usually "
            "means LoRA/helper/text-encoder params are present; disable BAdam or set "
            "badam_allow_unmatched_params=true to keep them always active."
        )
    if unmatched_count and logger is not None:
        logger.warning(
            "BAdam: %d trainable parameter(s) are not named by the transformer and "
            "will remain always active with synthetic names.",
            unmatched_count,
        )
    if unmatched_count and allow_unmatched:
        matched_ids = {id(param) for _name, param in named_parameters}
        unmatched_params = [
            param for param in flat_trainable if id(param) not in matched_ids
        ]
        named_parameters.extend(
            (f"badam_unmatched.{index}", param)
            for index, param in enumerate(unmatched_params)
        )
    if not named_parameters:
        raise ValueError(
            "BAdam found no trainable transformer parameters. It is intended for "
            "full or partial transformer fine-tuning, not pure LoRA parameter sets."
        )

    configured_prefixes = normalize_block_prefixes(
        getattr(args, "badam_block_prefixes", []) or []
    )
    prefix_mode = str(getattr(args, "badam_block_prefix_mode", "wan_blocks")).lower()
    if configured_prefixes:
        block_prefixes = configured_prefixes
    elif prefix_mode in {"auto", "wan_blocks"}:
        block_prefixes = infer_transformer_block_prefixes(
            named_parameters,
            include_embedding=bool(getattr(args, "badam_include_embedding", False)),
            include_lm_head=bool(getattr(args, "badam_include_lm_head", False)),
        )
    else:
        block_prefixes = []
    if not block_prefixes:
        raise ValueError(
            "BAdam could not infer any Wan block prefixes from trainable transformer "
            "parameters. Set badam_block_prefixes explicitly."
        )

    wrapper = TakenokoBlockOptimizer(
        base_optimizer,
        named_parameters,
        block_prefixes,
        switch_block_every=int(getattr(args, "badam_switch_block_every", 100)),
        switch_mode=str(getattr(args, "badam_switch_mode", "random")).lower(),
        start_block=getattr(args, "badam_start_block", None),
        always_active_prefixes=(
            list(getattr(args, "badam_always_active_prefixes", []) or [])
            + list(getattr(args, "badam_active_modules", []) or [])
        ),
        include_non_block=bool(getattr(args, "badam_include_non_block", True)),
        use_fp32_active_copy=bool(getattr(args, "badam_use_fp32_active_copy", True)),
        purge_inactive_state=bool(getattr(args, "badam_purge_inactive_state", True)),
        reset_state_on_switch=bool(getattr(args, "badam_reset_state_on_switch", True)),
        verbose=int(getattr(args, "badam_verbose", 1)),
        logger=logger,
    )
    if logger is not None:
        logger.info(
            "BAdam wrapped %s with %d block(s), switch_every=%d, mode=%s.",
            base_optimizer.__class__.__name__,
            len(block_prefixes),
            wrapper.switch_block_every,
            wrapper.switch_mode,
        )
    return wrapper


def create_badam_ratio_optimizer(
    args: Any,
    transformer: torch.nn.Module,
    trainable_params: Sequence[Any],
    *,
    distributed_context: dict[str, Any] | None = None,
    logger: Any | None = None,
) -> TakenokoBlockOptimizerRatio:
    """Create the sparse ratio-update BAdam variant."""

    distributed_context = distributed_context or {}
    world_size = int(distributed_context.get("world_size", 1) or 1)
    if world_size > 1 and not bool(getattr(args, "badam_allow_distributed", False)):
        raise ValueError(
            "BAdamRatio sparse-gradient hooks are currently supported only for "
            "single-process training in Takenoko."
        )
    if float(getattr(args, "max_grad_norm", 0.0)) != 0.0 and logger is not None:
        logger.warning(
            "BAdamRatio emits sparse gradients; set max_grad_norm=0 if your "
            "Accelerate/PyTorch build cannot clip sparse tensors."
        )

    flat_trainable = flatten_optimizer_params(list(trainable_params))
    trainable_ids = {id(param) for param in flat_trainable}
    named_parameters = [
        (name, param)
        for name, param in transformer.named_parameters()
        if id(param) in trainable_ids
    ]
    if not named_parameters:
        raise ValueError("BAdamRatio found no trainable transformer parameters.")

    optimizer = TakenokoBlockOptimizerRatio(
        trainable_params,
        named_parameters,
        update_ratio=float(getattr(args, "badam_update_ratio", 0.1)),
        switch_every=int(getattr(args, "badam_switch_block_every", 100)),
        preserve_threshold=int(getattr(args, "badam_ratio_preserve_threshold", 100)),
        mask_mode=str(getattr(args, "badam_ratio_mask_mode", "adjacent")).lower(),
        lr=float(getattr(args, "learning_rate", 1e-5)),
        keep_mask=bool(getattr(args, "badam_ratio_keep_mask", True)),
        include_embedding=bool(getattr(args, "badam_include_embedding", False)),
        include_lm_head=bool(getattr(args, "badam_include_lm_head", False)),
        verbose=int(getattr(args, "badam_verbose", 1)),
    )
    if logger is not None:
        logger.info(
            "BAdamRatio created with update_ratio=%.4f, switch_every=%d, mask_mode=%s.",
            optimizer.update_ratio,
            optimizer.switch_every,
            optimizer.mask_mode,
        )
    return optimizer
