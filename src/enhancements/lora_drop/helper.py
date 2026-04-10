from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


class LoRADropController:
    def __init__(
        self,
        *,
        enabled: bool,
        importance_ema_decay: float = 0.95,
        apply_after_steps: int = 500,
        apply_interval: int = 500,
        share_fraction: float = 0.25,
        min_group_size: int = 2,
    ) -> None:
        self.enabled = bool(enabled)
        self.importance_ema_decay = float(importance_ema_decay)
        self.apply_after_steps = int(apply_after_steps)
        self.apply_interval = int(apply_interval)
        self.share_fraction = float(share_fraction)
        self.min_group_size = int(min_group_size)
        self.step = 0
        self.last_newly_shared = 0
        self.last_apply_step = -1

    def build_module_kwargs(self) -> Dict[str, Any]:
        return {
            "lora_drop_enabled": self.enabled,
            "lora_drop_importance_ema_decay": self.importance_ema_decay,
        }

    def _collect_modules(self, network: Any) -> List[Any]:
        return [
            lora
            for lora in getattr(network, "text_encoder_loras", [])
            + getattr(network, "unet_loras", [])
            if hasattr(lora, "get_lora_drop_group_key")
            and hasattr(lora, "get_lora_drop_importance")
            and hasattr(lora, "set_lora_drop_shared_source")
        ]

    def apply_sharing(self, network: Any) -> int:
        if not self.enabled or self.share_fraction <= 0.0:
            return 0

        groups: Dict[tuple, List[Any]] = {}
        for lora in self._collect_modules(network):
            groups.setdefault(lora.get_lora_drop_group_key(), []).append(lora)

        newly_shared = 0
        for group_modules in groups.values():
            if len(group_modules) < self.min_group_size:
                continue

            shared_modules = [
                module
                for module in group_modules
                if module._get_lora_drop_shared_source() is not None
            ]
            independent_modules = [
                module
                for module in group_modules
                if getattr(module, "enabled", True)
                and module._get_lora_drop_shared_source() is None
            ]
            if len(independent_modules) < 2:
                continue

            target_shared = int(math.floor(len(group_modules) * self.share_fraction))
            target_shared = min(target_shared, len(group_modules) - 1)
            remaining_to_share = target_shared - len(shared_modules)
            if remaining_to_share <= 0:
                continue

            ranked_modules = sorted(
                independent_modules,
                key=lambda module: module.get_lora_drop_importance(),
                reverse=True,
            )
            anchor_module = ranked_modules[0]
            share_candidates = ranked_modules[1:]
            if not share_candidates:
                continue

            share_count = min(remaining_to_share, len(share_candidates))
            for module in share_candidates[-share_count:]:
                module.set_lora_drop_shared_source(anchor_module)
                newly_shared += 1

        return newly_shared

    def get_metrics(self, network: Any) -> Dict[str, float]:
        if not self.enabled:
            return {}

        modules = self._collect_modules(network)
        if not modules:
            return {}

        importances = [module.get_lora_drop_importance() for module in modules]
        shared_count = sum(
            1 for module in modules if module._get_lora_drop_shared_source() is not None
        )
        independent_count = sum(
            1
            for module in modules
            if getattr(module, "enabled", True)
            and module._get_lora_drop_shared_source() is None
        )
        total_count = len(modules)

        return {
            "lora_drop/importance_mean": float(sum(importances) / total_count),
            "lora_drop/importance_max": float(max(importances)),
            "lora_drop/importance_min": float(min(importances)),
            "lora_drop/shared_modules": float(shared_count),
            "lora_drop/shared_ratio": float(shared_count / total_count),
            "lora_drop/independent_ratio": float(independent_count / total_count),
            "lora_drop/last_newly_shared": float(self.last_newly_shared),
        }

    def on_step_start(self, network: Any, logger: Any) -> None:
        if not self.enabled:
            return

        self.step += 1
        self.last_newly_shared = 0

        if self.step < self.apply_after_steps:
            return
        if (self.step - self.apply_after_steps) % max(1, self.apply_interval) != 0:
            return

        self.last_newly_shared = self.apply_sharing(network)
        self.last_apply_step = self.step
        if self.last_newly_shared > 0:
            logger.info(
                "LoRA-drop shared %s low-importance adapters at step %s.",
                self.last_newly_shared,
                self.step,
            )

    def sync_shared_weights_for_export(self, network: Any) -> None:
        if not self.enabled:
            return
        for lora in getattr(network, "text_encoder_loras", []) + getattr(
            network, "unet_loras", []
        ):
            source = getattr(lora, "_get_lora_drop_shared_source", lambda: None)()
            if source is None:
                continue
            for target_param, source_param in zip(lora.parameters(), source.parameters()):
                target_param.data.copy_(source_param.data)


def create_lora_drop_controller(
    *,
    enabled: bool,
    importance_ema_decay: float = 0.95,
    apply_after_steps: int = 500,
    apply_interval: int = 500,
    share_fraction: float = 0.25,
    min_group_size: int = 2,
) -> Optional[LoRADropController]:
    if not enabled:
        return None
    return LoRADropController(
        enabled=enabled,
        importance_ema_decay=importance_ema_decay,
        apply_after_steps=apply_after_steps,
        apply_interval=apply_interval,
        share_fraction=share_fraction,
        min_group_size=min_group_size,
    )
