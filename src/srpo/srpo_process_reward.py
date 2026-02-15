"""
Process reward model integration for SRPO Euphonium mode.

Provides a pluggable interface for latent process reward models that can return
both scalar rewards and reward gradients with respect to noisy latents.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import importlib
import inspect
from typing import Any, Dict, Optional, Tuple

import torch


def _extract_score_tensor(output: Any, batch_size: int) -> torch.Tensor:
    if isinstance(output, dict):
        for key in ("score", "scores", "reward", "rewards", "logits"):
            value = output.get(key)
            if torch.is_tensor(value):
                output = value
                break
        else:
            raise ValueError(
                "Process reward model returned dict without tensor score key "
                "(expected one of: score/scores/reward/rewards/logits)."
            )
    elif isinstance(output, (tuple, list)):
        tensor_value = None
        for item in output:
            if torch.is_tensor(item):
                tensor_value = item
                break
        if tensor_value is None:
            raise ValueError(
                "Process reward model returned tuple/list without tensor output."
            )
        output = tensor_value

    if not torch.is_tensor(output):
        raise ValueError(
            f"Process reward model output must be tensor/dict/tuple, got {type(output)!r}"
        )

    scores = output
    if scores.ndim == 0:
        scores = scores.expand(batch_size)
    elif scores.shape[0] != batch_size:
        if scores.numel() == batch_size:
            scores = scores.reshape(batch_size)
        else:
            raise ValueError(
                "Process reward model output batch mismatch: expected first dim "
                f"{batch_size}, got shape {tuple(scores.shape)}."
            )
    elif scores.ndim > 1:
        scores = scores.reshape(batch_size, -1).mean(dim=1)
    return scores


class BaseSRPOProcessRewardModel(ABC):
    def __init__(self, device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype

    @abstractmethod
    def _forward_scores(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: Optional[torch.Tensor],
        prompt_attention_mask: Optional[torch.Tensor],
        pooled_prompt_embeds: Optional[torch.Tensor],
    ) -> torch.Tensor:
        raise NotImplementedError

    def compute_reward_and_gradient(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        latents = noisy_latents.detach().to(
            device=self.device, dtype=self.dtype
        ).requires_grad_(True)
        timesteps = timesteps.to(device=self.device)
        prompt_embeds = (
            prompt_embeds.to(device=self.device, dtype=self.dtype)
            if prompt_embeds is not None
            else None
        )
        prompt_attention_mask = (
            prompt_attention_mask.to(device=self.device)
            if prompt_attention_mask is not None
            else None
        )
        pooled_prompt_embeds = (
            pooled_prompt_embeds.to(device=self.device, dtype=self.dtype)
            if pooled_prompt_embeds is not None
            else None
        )

        scores = self._forward_scores(
            noisy_latents=latents,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            pooled_prompt_embeds=pooled_prompt_embeds,
        )
        self._validate_scores(scores=scores, batch_size=latents.shape[0])

        gradient = torch.autograd.grad(
            outputs=scores.sum(),
            inputs=latents,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
        return scores.detach(), gradient.detach()

    def compute_reward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        latents = noisy_latents.detach().to(device=self.device, dtype=self.dtype)
        timesteps = timesteps.to(device=self.device)
        prompt_embeds = (
            prompt_embeds.to(device=self.device, dtype=self.dtype)
            if prompt_embeds is not None
            else None
        )
        prompt_attention_mask = (
            prompt_attention_mask.to(device=self.device)
            if prompt_attention_mask is not None
            else None
        )
        pooled_prompt_embeds = (
            pooled_prompt_embeds.to(device=self.device, dtype=self.dtype)
            if pooled_prompt_embeds is not None
            else None
        )
        with torch.no_grad():
            scores = self._forward_scores(
                noisy_latents=latents,
                timesteps=timesteps,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                pooled_prompt_embeds=pooled_prompt_embeds,
            )
        self._validate_scores(scores=scores, batch_size=latents.shape[0])
        return scores.detach()

    @staticmethod
    def _validate_scores(scores: torch.Tensor, batch_size: int) -> None:
        if scores.ndim != 1 or scores.shape[0] != batch_size:
            raise ValueError(
                "Process reward scores must be 1D [B], got "
                f"shape {tuple(scores.shape)}."
            )


class TorchScriptProcessRewardModel(BaseSRPOProcessRewardModel):
    def __init__(self, checkpoint_path: str, device: torch.device, dtype: torch.dtype):
        super().__init__(device=device, dtype=dtype)
        self.checkpoint_path = checkpoint_path
        self.model = torch.jit.load(checkpoint_path, map_location=device)
        self.model.eval()
        if hasattr(self.model, "parameters"):
            for param in self.model.parameters():
                param.requires_grad_(False)

    def _forward_scores(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: Optional[torch.Tensor],
        prompt_attention_mask: Optional[torch.Tensor],
        pooled_prompt_embeds: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = noisy_latents.shape[0]
        forward_errors = []

        call_variants = [
            lambda: self.model(
                noisy_latents=noisy_latents,
                timesteps=timesteps,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                pooled_prompt_embeds=pooled_prompt_embeds,
            ),
            lambda: self.model(
                noisy_latents, timesteps, prompt_embeds, prompt_attention_mask, pooled_prompt_embeds
            ),
            lambda: self.model(noisy_latents, timesteps),
            lambda: self.model(noisy_latents),
        ]
        for variant in call_variants:
            try:
                output = variant()
                return _extract_score_tensor(output, batch_size=batch_size).to(
                    device=self.device, dtype=self.dtype
                )
            except Exception as exc:  # pragma: no cover - fallback path
                forward_errors.append(str(exc))

        raise RuntimeError(
            "Failed to execute TorchScript process reward model with supported call "
            f"signatures. Errors: {forward_errors}"
        )


class PythonCallableProcessRewardModel(BaseSRPOProcessRewardModel):
    def __init__(
        self,
        entry: str,
        checkpoint_path: str,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__(device=device, dtype=dtype)
        self.entry = entry
        self.checkpoint_path = checkpoint_path
        self.callable_obj = self._load_callable(entry, checkpoint_path, device, dtype)

    @staticmethod
    def _load_callable(
        entry: str,
        checkpoint_path: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Any:
        if ":" in entry:
            module_name, attr_name = entry.split(":", 1)
        else:
            module_name, attr_name = entry.rsplit(".", 1)
        module = importlib.import_module(module_name)
        target = getattr(module, attr_name)

        if inspect.isclass(target):
            init_attempts = [
                lambda: target(
                    checkpoint_path=checkpoint_path,
                    device=device,
                    dtype=dtype,
                ),
                lambda: target(checkpoint_path=checkpoint_path),
                lambda: target(device=device, dtype=dtype),
                lambda: target(),
            ]
            errors = []
            for attempt in init_attempts:
                try:
                    return attempt()
                except Exception as exc:  # pragma: no cover - fallback path
                    errors.append(str(exc))
            raise RuntimeError(
                f"Failed to construct process reward class '{entry}'. Errors: {errors}"
            )

        if callable(target):
            return target

        raise TypeError(f"Process reward entry '{entry}' is not callable.")

    def _forward_scores(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: Optional[torch.Tensor],
        prompt_attention_mask: Optional[torch.Tensor],
        pooled_prompt_embeds: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = noisy_latents.shape[0]
        kwargs = {
            "noisy_latents": noisy_latents,
            "timesteps": timesteps,
            "prompt_embeds": prompt_embeds,
            "prompt_attention_mask": prompt_attention_mask,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "checkpoint_path": self.checkpoint_path,
        }

        errors = []
        for fn in (
            lambda: self.callable_obj(**kwargs),
            lambda: self.callable_obj(noisy_latents, timesteps, prompt_embeds),
            lambda: self.callable_obj(noisy_latents, timesteps),
            lambda: self.callable_obj(noisy_latents),
        ):
            try:
                output = fn()
                return _extract_score_tensor(output, batch_size=batch_size).to(
                    device=self.device, dtype=self.dtype
                )
            except Exception as exc:  # pragma: no cover - fallback path
                errors.append(str(exc))

        raise RuntimeError(
            f"Failed to execute python callable process reward model '{self.entry}'. "
            f"Errors: {errors}"
        )


def create_process_reward_model(
    *,
    model_type: str,
    model_path: str,
    model_entry: str,
    model_dtype: str,
    device: torch.device,
    logger: Any,
) -> Optional[BaseSRPOProcessRewardModel]:
    model_type = model_type.lower()
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    dtype = dtype_map.get(model_dtype.lower(), torch.float32)

    if model_type == "none":
        return None
    if model_type == "torchscript":
        logger.info(
            "Loading SRPO process reward TorchScript model from: %s (dtype=%s)",
            model_path,
            model_dtype,
        )
        return TorchScriptProcessRewardModel(
            checkpoint_path=model_path,
            device=device,
            dtype=dtype,
        )
    if model_type == "python_callable":
        logger.info(
            "Loading SRPO process reward python callable: %s (dtype=%s, checkpoint=%s)",
            model_entry,
            model_dtype,
            model_path if model_path else "<none>",
        )
        return PythonCallableProcessRewardModel(
            entry=model_entry,
            checkpoint_path=model_path,
            device=device,
            dtype=dtype,
        )
    raise ValueError(f"Unsupported process reward model_type: {model_type!r}")
