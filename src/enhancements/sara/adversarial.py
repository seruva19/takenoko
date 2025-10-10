"""Adversarial alignment utilities for SARA."""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.logger import get_logger

from .config import SaraConfig
from .utils import gradient_penalty

try:
    from torchvision import models
    from torchvision.models import ResNet18_Weights
except Exception:  # pragma: no cover - torchvision is optional
    models = None
    ResNet18_Weights = None


logger = get_logger(__name__)


class _MLPResidualBlock(nn.Module):
    """Simple residual block for lightweight MLP discriminators."""

    def __init__(self, dim: int, out_dim: Optional[int] = None) -> None:
        super().__init__()
        out_dim = out_dim or dim
        self.linear1 = nn.Linear(dim, out_dim)
        self.norm1 = nn.LayerNorm(out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.skip = (
            nn.Identity()
            if dim == out_dim
            else nn.Sequential(nn.Linear(dim, out_dim), nn.LayerNorm(out_dim))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.linear1(x)
        out = self.activation(self.norm1(out))
        out = self.linear2(out)
        out = self.norm2(out)
        out = self.activation(out + residual)
        return out


class MLPDiscriminator(nn.Module):
    """Compact discriminator that operates on averaged token embeddings."""

    def __init__(self, input_dim: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(
            _MLPResidualBlock(hidden_dim),
            _MLPResidualBlock(hidden_dim),
            _MLPResidualBlock(hidden_dim, hidden_dim // 2),
        )
        self.head = nn.Linear(hidden_dim // 2, 1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.mean(dim=1)
        x = self.activation(self.input_proj(x))
        x = self.blocks(x)
        return self.head(x)


class SimpleCNNDiscriminator(nn.Module):
    """Fallback discriminator for lower-resource runs."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(256, 128, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = x.transpose(1, 2)
        x = self.net(x).squeeze(-1)
        return self.head(x)


class ResNet18Discriminator(nn.Module):
    """Truncated ResNet-18 discriminator with 1x1 stem as described in the paper."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        if models is None or ResNet18_Weights is None:
            raise ImportError(
                "torchvision is required for the ResNet-18 discriminator; "
                "please install torchvision or select a different discriminator_arch."
            )

        backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # We borrow the ImageNet backbone but immediately replace the 7x7 stem with
        # a 1x1 projection that averages the pretrained filters. This keeps the
        # deeper residual blocks (which provide a useful inductive bias) while
        # discarding the RGB-specific assumptions that worried earlier reviewers.

        self.input_proj = nn.Conv2d(input_dim, 64, kernel_size=1, stride=1, bias=False)
        with torch.no_grad():
            conv1_weight = backbone.conv1.weight
            base_weight = conv1_weight.mean(dim=(2, 3), keepdim=True).mean(
                dim=1, keepdim=True
            )
            expanded = base_weight.repeat(1, input_dim, 1, 1) / max(1, input_dim)
            self.input_proj.weight.copy_(expanded)

        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = nn.Identity()
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(128, 1)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            bsz, tokens, dim = x.shape
            height = max(1, int(math.isqrt(tokens)))
            width = max(1, math.ceil(tokens / height))
            total = height * width
            if total != tokens:
                pad = total - tokens
                if pad > 0:
                    x = torch.cat(
                        [x, x.new_zeros(bsz, pad, dim)],
                        dim=1,
                    )
            x = x.transpose(1, 2).contiguous().view(bsz, dim, height, width)
        elif x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        else:
            raise ValueError(f"Unsupported feature shape for discriminator: {x.shape}")

        x = self.input_proj(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.head(x)


class AdversarialAligner(nn.Module):
    """Discriminator-based representation alignment."""

    def __init__(
        self,
        config: SaraConfig,
        feature_dim: int,
        autocast_factory: Optional[Callable[[], torch.autocast]] = None,
        grad_scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.feature_dim = feature_dim
        self.current_step = 0

        if config.discriminator_arch == "resnet18":
            self.discriminator = ResNet18Discriminator(feature_dim)
        elif config.discriminator_arch in {"mlp", "resnet18_mlp"}:
            self.discriminator = MLPDiscriminator(feature_dim)
        elif config.discriminator_arch == "simple_cnn":
            self.discriminator = SimpleCNNDiscriminator(feature_dim)
        else:
            raise ValueError(
                f"Unsupported discriminator architecture: {config.discriminator_arch}"
            )

        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config.discriminator_lr,
            betas=(0.5, 0.999),
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self._amp_enabled = config.use_mixed_precision and torch.cuda.is_available()
        self._autocast_factory: Optional[Callable[[], torch.autocast]] = autocast_factory
        self._external_scaler = grad_scaler is not None
        self._max_grad_norm = config.discriminator_max_grad_norm
        self._log_detailed_metrics = config.log_detailed_metrics
        self._feature_matching_weight = config.feature_matching_weight
        if grad_scaler is not None:
            self.scaler: Optional[torch.cuda.amp.GradScaler] = grad_scaler
        else:
            self.scaler = (
                torch.cuda.amp.GradScaler(enabled=self._amp_enabled)
                if self._amp_enabled
                else None
            )
        if config.discriminator_scheduler_step > 0:
            self.scheduler: Optional[torch.optim.lr_scheduler.StepLR] = torch.optim.lr_scheduler.StepLR(
                self.discriminator_optimizer,
                step_size=config.discriminator_scheduler_step,
                gamma=config.discriminator_scheduler_gamma,
            )
        else:
            self.scheduler = None

        logger.info(
            "SARA adversarial aligner: arch=%s lr=%.2e warmup=%d",
            config.discriminator_arch,
            config.discriminator_lr,
            config.discriminator_warmup_steps,
        )

    def configure_mixed_precision(
        self,
        autocast_factory: Optional[Callable[[], torch.autocast]] = None,
        grad_scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ) -> None:
        """Allow external accelerators to manage autocast/GradScaler."""
        if autocast_factory is not None:
            self._autocast_factory = autocast_factory
        if grad_scaler is not None:
            self.scaler = grad_scaler
            self._external_scaler = True
        elif not self._external_scaler and self._amp_enabled:
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)

    def _get_autocast_context(self):
        if not self._amp_enabled and self._autocast_factory is None:
            return nullcontext()
        if self._autocast_factory is not None:
            return self._autocast_factory()
        return torch.cuda.amp.autocast(enabled=self._amp_enabled)

    def update_discriminator(
        self,
        real_features: torch.Tensor,
        fake_features: torch.Tensor,
    ) -> Dict[str, float]:
        self.discriminator.train()
        device = real_features.device
        batch_size = real_features.size(0)
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        self.discriminator_optimizer.zero_grad(set_to_none=True)

        grad_norm: Optional[float] = None
        with self._get_autocast_context():
            real_output = self.discriminator(real_features.detach())
            fake_output = self.discriminator(fake_features.detach())
            d_loss_real = self.criterion(real_output, real_labels)
            d_loss_fake = self.criterion(fake_output, fake_labels)
            d_loss = d_loss_real + d_loss_fake

            if self.config.gradient_penalty_weight > 0:
                gp = gradient_penalty(
                    self.discriminator,
                    real_features.detach(),
                    fake_features.detach(),
                    device=device,
                )
                d_loss = d_loss + self.config.gradient_penalty_weight * gp
            else:
                gp = torch.tensor(0.0, device=device)

        if self.scaler is not None:
            self.scaler.scale(d_loss).backward()
            if self._max_grad_norm is not None or self._log_detailed_metrics:
                self.scaler.unscale_(self.discriminator_optimizer)
            if self._max_grad_norm is not None:
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        self.discriminator.parameters(),
                        self._max_grad_norm,
                    ).item()
                )
            elif self._log_detailed_metrics:
                grad_norm = self._compute_grad_norm()
            self.scaler.step(self.discriminator_optimizer)
            self.scaler.update()
        else:
            d_loss.backward()
            if self._max_grad_norm is not None:
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        self.discriminator.parameters(),
                        self._max_grad_norm,
                    ).item()
                )
            elif self._log_detailed_metrics:
                grad_norm = self._compute_grad_norm()
            self.discriminator_optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        with torch.no_grad():
            d_real = torch.sigmoid(real_output)
            d_fake = torch.sigmoid(fake_output)
            real_acc = (d_real > 0.5).float().mean().item()
            fake_acc = (d_fake < 0.5).float().mean().item()

        metrics = {
            "d_loss": float(d_loss.detach().item()),
            "d_loss_real": float(d_loss_real.detach().item()),
            "d_loss_fake": float(d_loss_fake.detach().item()),
            "d_gradient_penalty": float(gp.detach().item()),
            "d_accuracy": (real_acc + fake_acc) / 2.0,
            "d_real_acc": real_acc,
            "d_fake_acc": fake_acc,
        }
        if grad_norm is not None:
            metrics["d_grad_norm"] = grad_norm
        if self._log_detailed_metrics:
            metrics["d_lr"] = float(self.discriminator_optimizer.param_groups[0]["lr"])
        return metrics

    def compute_generator_loss(
        self,
        fake_features: torch.Tensor,
        real_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        self.discriminator.eval()
        if not self.is_past_warmup():
            return fake_features.new_tensor(0.0), {"g_adv_loss": 0.0}

        batch_size = fake_features.size(0)
        device = fake_features.device
        real_labels = torch.ones(batch_size, 1, device=device)

        with self._get_autocast_context():
            fake_output = self.discriminator(fake_features)
            g_loss = self.criterion(fake_output, real_labels)

            if self.config.feature_matching and real_features is not None:
                fake_mean = fake_features.mean(dim=1)
                real_mean = real_features.mean(dim=1)
                fm_loss = F.l1_loss(fake_mean, real_mean)
                if self._feature_matching_weight > 0:
                    g_loss = g_loss + self._feature_matching_weight * fm_loss
            else:
                fm_loss = fake_features.new_tensor(0.0)

        metrics = {
            "g_adv_loss": float(g_loss.detach().item()),
            "g_adv_accuracy": float(
                (torch.sigmoid(fake_output) > 0.5).float().mean().item()
            ),
            "g_feature_matching": float(fm_loss.detach().item()),
        }
        if self._log_detailed_metrics:
            metrics["g_logits_mean"] = float(fake_output.detach().mean().item())
            metrics["g_logits_std"] = float(fake_output.detach().std().item())
        return g_loss, metrics

    def increment_step(self) -> None:
        self.current_step += 1

    def is_past_warmup(self) -> bool:
        return self.current_step >= self.config.discriminator_warmup_steps

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        state = super().state_dict(*args, **kwargs)
        state["discriminator_optimizer"] = self.discriminator_optimizer.state_dict()
        state["current_step"] = self.current_step
        if self.scheduler is not None:
            state["discriminator_scheduler"] = self.scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]  # noqa: ANN001
        optimizer_state = state_dict.pop("discriminator_optimizer", None)
        self.current_step = state_dict.pop("current_step", 0)
        scheduler_state = state_dict.pop("discriminator_scheduler", None)
        super().load_state_dict(state_dict, strict=strict)
        if optimizer_state is not None:
            self.discriminator_optimizer.load_state_dict(optimizer_state)
        if self.scheduler is not None and scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)

    def _compute_grad_norm(self) -> float:
        total = 0.0
        for param in self.discriminator.parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            total += float(grad.norm(2).item() ** 2)
        return math.sqrt(total)
