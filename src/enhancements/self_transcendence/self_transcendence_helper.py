"""Self-Transcendence helper for VAE alignment + self-guided feature supervision."""

from __future__ import annotations

import copy
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.logger import get_logger

logger = get_logger(__name__)


class SelfTranscendenceHelper(nn.Module):
    """Compute Self-Transcendence alignment losses with lightweight MLP heads."""

    def __init__(self, transformer: nn.Module, args: Any, model_config: Any) -> None:
        super().__init__()
        self.args = args
        self.model_config = model_config
        self.transformer = self._unwrap_model(transformer)
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.teacher_hooks: List[torch.utils.hooks.RemovableHandle] = []

        self.student_guided_features: Optional[torch.Tensor] = None
        self.teacher_guiding_cond: Optional[torch.Tensor] = None
        self.teacher_guiding_uncond: Optional[torch.Tensor] = None
        self._teacher_capture_mode: Optional[str] = None

        self.teacher: Optional[nn.Module] = None
        self._teacher_ready: bool = False
        self._warned_teacher_failure: bool = False

        self.guided_layer, self.guiding_layer = self._resolve_layer_indices()

        self.hidden_dim = self._resolve_hidden_dim()
        mlp_mult = int(getattr(args, "self_transcendence_mlp_multiplier", 2))
        if mlp_mult < 1:
            mlp_mult = 1

        self.vae_latent_dim = int(
            getattr(model_config, "out_dim", 16) or 16
        )

        self.vae_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * mlp_mult),
            nn.SiLU(),
            nn.Linear(self.hidden_dim * mlp_mult, self.vae_latent_dim),
        )
        self.feature_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * mlp_mult),
            nn.SiLU(),
            nn.Linear(self.hidden_dim * mlp_mult, self.hidden_dim),
        )

    @staticmethod
    def _unwrap_model(model: nn.Module) -> nn.Module:
        return model.module if hasattr(model, "module") else model

    def _resolve_hidden_dim(self) -> int:
        if hasattr(self.transformer, "dim"):
            return int(self.transformer.dim)
        if hasattr(self.transformer, "hidden_size"):
            return int(self.transformer.hidden_size)
        for module in self.transformer.modules():
            if hasattr(module, "in_features"):
                return int(module.in_features)
        logger.warning("Self-Transcendence: falling back to hidden_dim=1024")
        return 1024

    def _get_blocks(self) -> Tuple[List[nn.Module], int]:
        blocks = getattr(self.transformer, "blocks", None)
        if blocks is None and hasattr(self.transformer, "module"):
            blocks = getattr(self.transformer.module, "blocks", None)
        if blocks is None:
            raise ValueError("Self-Transcendence: transformer blocks not found")
        return list(blocks), len(blocks)

    def _resolve_layer_indices(self) -> Tuple[int, int]:
        try:
            _, num_blocks = self._get_blocks()
        except Exception:
            num_blocks = 0

        guided = int(getattr(self.args, "self_transcendence_guided_layer", -1))
        guiding = int(getattr(self.args, "self_transcendence_guiding_layer", -1))

        if num_blocks <= 0:
            return max(1, guided), max(1, guiding)

        if guided <= 0:
            guided = max(1, num_blocks // 2)
        if guiding <= 0:
            guiding = max(guided + 1, (2 * num_blocks) // 3)
        if guiding <= guided:
            guiding = min(num_blocks, guided + 1)
        return guided, guiding

    @staticmethod
    def _resolve_block_index(idx: int, num_blocks: int) -> Optional[int]:
        candidates = [idx - 1, idx]
        for cand in candidates:
            if 0 <= cand < num_blocks:
                return cand + 1
        return None

    def _make_student_hook(self, block_idx: int):
        def hook(_module, _inputs, output):
            tensor = output
            if isinstance(output, (list, tuple)) and output:
                tensor = output[0]
            if torch.is_tensor(tensor):
                self.student_guided_features = tensor

        return hook

    def _make_teacher_hook(self, block_idx: int):
        def hook(_module, _inputs, output):
            tensor = output
            if isinstance(output, (list, tuple)) and output:
                tensor = output[0]
            if not torch.is_tensor(tensor):
                return
            if self._teacher_capture_mode == "cond":
                self.teacher_guiding_cond = tensor
            elif self._teacher_capture_mode == "uncond":
                self.teacher_guiding_uncond = tensor

        return hook

    def setup_hooks(self) -> None:
        self.remove_hooks()
        blocks, num_blocks = self._get_blocks()
        guided_idx = self._resolve_block_index(self.guided_layer, num_blocks)
        if guided_idx is None:
            raise ValueError(
                f"Self-Transcendence: guided_layer {self.guided_layer} out of range"
            )
        handle = blocks[guided_idx - 1].register_forward_hook(
            self._make_student_hook(guided_idx)
        )
        self.hooks.append(handle)
        logger.info(
            "Self-Transcendence: hooked guided layer %d (num_blocks=%d)",
            guided_idx,
            num_blocks,
        )

    def _setup_teacher_hooks(self) -> None:
        for handle in self.teacher_hooks:
            try:
                handle.remove()
            except Exception:
                pass
        self.teacher_hooks = []
        if self.teacher is None:
            return
        try:
            blocks, num_blocks = self._get_teacher_blocks()
        except Exception as exc:
            logger.warning("Self-Transcendence: teacher blocks not found: %s", exc)
            return
        guiding_idx = self._resolve_block_index(self.guiding_layer, num_blocks)
        if guiding_idx is None:
            logger.warning(
                "Self-Transcendence: guiding_layer %s out of range for teacher",
                self.guiding_layer,
            )
            return
        handle = blocks[guiding_idx - 1].register_forward_hook(
            self._make_teacher_hook(guiding_idx)
        )
        self.teacher_hooks.append(handle)

    def _get_teacher_blocks(self) -> Tuple[List[nn.Module], int]:
        assert self.teacher is not None
        blocks = getattr(self.teacher, "blocks", None)
        if blocks is None and hasattr(self.teacher, "module"):
            blocks = getattr(self.teacher.module, "blocks", None)
        if blocks is None:
            raise ValueError("Self-Transcendence: teacher blocks not found")
        return list(blocks), len(blocks)

    def remove_hooks(self) -> None:
        for handle in self.hooks:
            try:
                handle.remove()
            except Exception:
                pass
        for handle in self.teacher_hooks:
            try:
                handle.remove()
            except Exception:
                pass
        self.hooks = []
        self.teacher_hooks = []
        self.student_guided_features = None
        self.teacher_guiding_cond = None
        self.teacher_guiding_uncond = None
        self._teacher_capture_mode = None

    def get_trainable_params(self) -> List[nn.Parameter]:
        params = list(self.vae_projection.parameters())
        params.extend(self.feature_projection.parameters())
        return params

    def _use_epoch_schedule(self, current_epoch: Optional[Any]) -> bool:
        warmup_epochs = int(getattr(self.args, "self_transcendence_warmup_epochs", 0))
        guidance_epochs = int(
            getattr(self.args, "self_transcendence_guidance_epochs", 0)
        )
        return (
            current_epoch is not None and (warmup_epochs > 0 or guidance_epochs > 0)
        )

    def _get_epoch_value(self, current_epoch: Optional[Any]) -> Optional[int]:
        if current_epoch is None:
            return None
        if hasattr(current_epoch, "value"):
            return int(current_epoch.value)
        try:
            return int(current_epoch)
        except Exception:
            return None

    def _stage_from_schedule(
        self, global_step: Optional[int], current_epoch: Optional[Any]
    ) -> str:
        if self._use_epoch_schedule(current_epoch):
            epoch = self._get_epoch_value(current_epoch)
            warmup_epochs = int(
                getattr(self.args, "self_transcendence_warmup_epochs", 0)
            )
            guidance_epochs = int(
                getattr(self.args, "self_transcendence_guidance_epochs", 0)
            )
            if epoch is None:
                return "disabled"
            if warmup_epochs > 0 and epoch <= warmup_epochs:
                return "vae"
            if guidance_epochs <= 0:
                return "self_guided"
            if epoch <= warmup_epochs + guidance_epochs:
                return "self_guided"
            return "disabled"

        warmup_steps = int(
            getattr(self.args, "self_transcendence_warmup_steps", 0)
        )
        guidance_steps = int(
            getattr(self.args, "self_transcendence_guidance_steps", 0)
        )
        if global_step is None:
            return "disabled"
        if warmup_steps > 0 and global_step < warmup_steps:
            return "vae"
        if guidance_steps <= 0:
            return "self_guided"
        if global_step < warmup_steps + guidance_steps:
            return "self_guided"
        return "disabled"

    def _maybe_build_teacher(self) -> None:
        if self.teacher is not None or self._teacher_ready:
            return
        try:
            self.teacher = copy.deepcopy(self.transformer)
            self.teacher.eval()
            self.teacher.requires_grad_(False)
            self._teacher_ready = True
            self._setup_teacher_hooks()
            logger.info("Self-Transcendence: teacher snapshot created")
        except Exception as exc:
            if not self._warned_teacher_failure:
                logger.warning(
                    "Self-Transcendence: failed to snapshot teacher model: %s",
                    exc,
                )
                self._warned_teacher_failure = True
            self.teacher = None
            self._teacher_ready = False

    def _build_context(
        self, batch: dict[str, Any], device: torch.device, dtype: torch.dtype
    ) -> List[torch.Tensor]:
        context_source = batch.get("t5")
        if context_source is None:
            return []
        return [t.to(device=device, dtype=dtype) for t in context_source]

    def _build_uncond_context(
        self, context: List[torch.Tensor], batch: dict[str, Any]
    ) -> List[torch.Tensor]:
        if not context:
            return []
        mode = str(getattr(self.args, "self_transcendence_uncond_mode", "zero")).lower()
        if mode == "batch_null" and "t5_uncond" in batch:
            return [
                t.to(device=context[0].device, dtype=context[0].dtype)
                for t in batch["t5_uncond"]
            ]
        return [torch.zeros_like(t) for t in context]

    def _vae_latent_tokens(
        self, latents: torch.Tensor, patch_size: Tuple[int, int, int]
    ) -> torch.Tensor:
        if latents.dim() == 5:
            pooled = F.avg_pool3d(latents, kernel_size=patch_size, stride=patch_size)
            tokens = pooled.flatten(2).transpose(1, 2)
        else:
            _, ph, pw = patch_size
            pooled = F.avg_pool2d(latents, kernel_size=(ph, pw), stride=(ph, pw))
            tokens = pooled.flatten(2).transpose(1, 2)
        return tokens

    def _align_token_counts(
        self, src: torch.Tensor, tgt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if src.shape[1] == tgt.shape[1]:
            return src, tgt
        min_tokens = min(src.shape[1], tgt.shape[1])
        return src[:, :min_tokens], tgt[:, :min_tokens]

    def _align_feature_dims(
        self, src: torch.Tensor, tgt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if src.shape[-1] == tgt.shape[-1]:
            return src, tgt
        if src.shape[-1] > tgt.shape[-1]:
            src = src[..., : tgt.shape[-1]]
        else:
            pad = tgt.shape[-1] - src.shape[-1]
            src = F.pad(src, (0, pad))
        return src, tgt

    def _run_teacher_forward(
        self,
        *,
        model_input: torch.Tensor,
        timesteps: torch.Tensor,
        context: List[torch.Tensor],
        seq_len: int,
    ) -> Optional[torch.Tensor]:
        if self.teacher is None:
            return None
        self._teacher_capture_mode = "cond"
        with torch.no_grad():
            _ = self.teacher(
                model_input,
                t=timesteps,
                context=context,
                seq_len=seq_len,
                y=None,
            )
        return self.teacher_guiding_cond

    def _run_teacher_forward_uncond(
        self,
        *,
        model_input: torch.Tensor,
        timesteps: torch.Tensor,
        context: List[torch.Tensor],
        seq_len: int,
    ) -> Optional[torch.Tensor]:
        if self.teacher is None:
            return None
        self._teacher_capture_mode = "uncond"
        with torch.no_grad():
            _ = self.teacher(
                model_input,
                t=timesteps,
                context=context,
                seq_len=seq_len,
                y=None,
            )
        return self.teacher_guiding_uncond

    def compute_loss(
        self,
        *,
        accelerator: Any,
        latents: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
        batch: dict[str, Any],
        global_step: Optional[int],
        current_epoch: Optional[Any],
    ) -> Optional[torch.Tensor]:
        if not getattr(self.args, "enable_self_transcendence", False):
            return None

        stage = self._stage_from_schedule(global_step, current_epoch)
        if stage == "disabled":
            self.student_guided_features = None
            return None

        if self.student_guided_features is None:
            return None

        patch_size = tuple(getattr(self.model_config, "patch_size", (1, 1, 1)))
        if len(patch_size) != 3:
            patch_size = (1, 1, 1)

        loss: Optional[torch.Tensor] = None

        if stage == "vae":
            target_tokens = self._vae_latent_tokens(latents, patch_size)
            projected = self.vae_projection(self.student_guided_features)
            projected, target_tokens = self._align_token_counts(
                projected, target_tokens
            )
            projected, target_tokens = self._align_feature_dims(
                projected, target_tokens
            )
            vae_loss = F.mse_loss(projected, target_tokens)
            vae_weight = float(
                getattr(self.args, "self_transcendence_vae_loss_weight", 1.0)
            )
            loss = vae_loss * vae_weight

        if stage == "self_guided":
            self._maybe_build_teacher()
            if self.teacher is None:
                return loss

            pt, ph, pw = patch_size
            if latents.dim() == 5:
                lat_f, lat_h, lat_w = latents.shape[2:5]
                seq_len = (lat_f * lat_h * lat_w) // (pt * ph * pw)
            else:
                lat_h, lat_w = latents.shape[2:4]
                seq_len = (lat_h * lat_w) // (ph * pw)

            context = self._build_context(
                batch, accelerator.device, network_dtype
            )
            uncond_context = self._build_uncond_context(context, batch)

            model_input = noisy_model_input.to(
                device=accelerator.device, dtype=network_dtype
            )
            timesteps = timesteps.to(device=accelerator.device)

            with accelerator.autocast():
                cond_feat = self._run_teacher_forward(
                    model_input=model_input,
                    timesteps=timesteps,
                    context=context,
                    seq_len=seq_len,
                )
                uncond_feat = self._run_teacher_forward_uncond(
                    model_input=model_input,
                    timesteps=timesteps,
                    context=uncond_context,
                    seq_len=seq_len,
                )

            if cond_feat is None or uncond_feat is None:
                return loss

            guidance_scale = float(
                getattr(self.args, "self_transcendence_guidance_scale", 30.0)
            )
            guided_target = uncond_feat + guidance_scale * (cond_feat - uncond_feat)
            guided_target = guided_target.detach()

            projected = self.feature_projection(self.student_guided_features)
            projected, guided_target = self._align_token_counts(
                projected, guided_target
            )
            projected, guided_target = self._align_feature_dims(
                projected, guided_target
            )
            guide_loss = F.mse_loss(projected, guided_target)
            guide_weight = float(
                getattr(self.args, "self_transcendence_guidance_loss_weight", 0.5)
            )
            guide_loss = guide_loss * guide_weight
            loss = guide_loss if loss is None else loss + guide_loss

        self.student_guided_features = None
        self.teacher_guiding_cond = None
        self.teacher_guiding_uncond = None
        self._teacher_capture_mode = None
        return loss
