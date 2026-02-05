from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .spec import LoRATargetSpec


class ResidualProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, scale: float = 0.05):
        super().__init__()
        self.scale = scale
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x) * self.scale


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int = 3):
        super().__init__()
        blocks: List[nn.Module] = []
        dim = input_dim
        for _ in range(layers):
            blocks.extend(
                [
                    nn.Linear(dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                ]
            )
            dim = hidden_dim
        self.net = nn.Sequential(*blocks)
        self.out_dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out


class PolyLoRAHead(nn.Module):
    def __init__(self, input_dim: int, spec: LoRATargetSpec, hidden_dim: int, enable_base_branch: bool = False):
        super().__init__()
        out_dim = spec.down_shape.numel() + spec.up_shape.numel()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.base_proj = None
        if enable_base_branch:
            self.base_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, out_dim),
            )
        self.spec = spec
        self.enable_base_branch = enable_base_branch

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        head_out = self.proj(x)
        down_flat = head_out[..., : self.spec.down_shape.numel()]
        up_flat = head_out[..., self.spec.down_shape.numel() :]
        down = down_flat.view(-1, *self.spec.down_shape)
        up = up_flat.view(-1, *self.spec.up_shape)
        alpha = torch.full((down.shape[0],), self.spec.down_shape[0], device=x.device, dtype=down.dtype)
        out = {
            f"{self.spec.name}.lora_down.weight": down,
            f"{self.spec.name}.lora_up.weight": up,
            f"{self.spec.name}.alpha": alpha,
        }
        if self.base_proj is not None:
            base_out = self.base_proj(x)
            b_down_flat = base_out[..., : self.spec.down_shape.numel()]
            b_up_flat = base_out[..., self.spec.down_shape.numel() :]
            b_down = b_down_flat.view(-1, *self.spec.down_shape)
            b_up = b_up_flat.view(-1, *self.spec.up_shape)
            out[f"base.{self.spec.name}.lora_down.weight"] = b_down
            out[f"base.{self.spec.name}.lora_up.weight"] = b_up
            out[f"base.{self.spec.name}.alpha"] = torch.full(
                (b_down.shape[0],), self.spec.down_shape[0], device=x.device, dtype=b_down.dtype
            )
        return out


class EmbeddingFusion(nn.Module):
    """Fuse style and identity embeddings with configurable strategy."""

    def __init__(self, style_dim: int, identity_dim: Optional[int] = None, mode: str = "style_only"):
        super().__init__()
        self.style_dim = style_dim
        self.identity_dim = identity_dim
        self.mode = mode
        self.identity_proj = None
        if mode != "style_only" and identity_dim is not None:
            self.identity_proj = nn.Linear(identity_dim, style_dim)
        self.gate = None
        if mode == "gated":
            self.gate = nn.Sequential(
                nn.Linear(style_dim * 2, style_dim),
                nn.SiLU(),
                nn.Linear(style_dim, style_dim),
                nn.Sigmoid(),
            )

    @property
    def output_dim(self) -> int:
        if self.mode == "concat" and self.identity_dim is not None:
            return self.style_dim * 2
        return self.style_dim

    def forward(self, style: torch.Tensor, identity: Optional[torch.Tensor]) -> torch.Tensor:
        if identity is None or self.mode == "style_only":
            return style
        if self.identity_proj is None:
            raise ValueError("identity_proj not initialized for fusion with identity.")
        id_proj = self.identity_proj(identity)
        if self.mode == "mean":
            return (style + id_proj) * 0.5
        if self.mode == "gated":
            gate = self.gate(torch.cat([style, id_proj], dim=-1)) if self.gate is not None else torch.sigmoid(style)
            return style * gate + id_proj * (1 - gate)
        if self.mode == "concat":
            return torch.cat([style, id_proj], dim=-1)
        raise ValueError(f"Unsupported fusion mode: {self.mode}")


class PerceiverMixer(nn.Module):
    """Lightweight Perceiver-style mixer over style/identity tokens."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 512,
        num_latents: int = 8,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim) / latent_dim**0.5)
        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, batch_first=True)
            ff = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, latent_dim * ff_mult),
                nn.SiLU(),
                nn.Linear(latent_dim * ff_mult, latent_dim),
            )
            self.layers.append(nn.ModuleDict({"attn": attn, "ff": ff, "ln": nn.LayerNorm(latent_dim)}))
        self.out_norm = nn.LayerNorm(latent_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, T, D]
        latents = self.latents.unsqueeze(0).repeat(tokens.size(0), 1, 1)
        inputs = self.input_proj(tokens)
        for layer in self.layers:
            attn_out, _ = layer["attn"](latents, inputs, inputs, need_weights=False)
            latents = layer["ln"](latents + attn_out)
            latents = latents + layer["ff"](latents)
        latents = self.out_norm(latents)
        # mean pool latents
        return latents.mean(dim=1)


class PolyLoRANetwork(nn.Module):
    """Hypernetwork that emits LoRA weights for a fixed spec list."""

    def __init__(
        self,
        embed_dim: int,
        target_specs: Iterable[LoRATargetSpec],
        trunk_hidden_dim: int = 512,
        trunk_layers: int = 2,
        head_hidden_dim: int = 512,
        use_residual: bool = True,
        residual_scale: float = 0.05,
        residual_dim: Optional[int] = None,
        normalization: bool = True,
        head_mode: str = "trunk",  # "trunk" (shared trunk + heads) or "per_tensor"
        fusion_mode: str = "style_only",  # style_only, mean, concat, gated
        identity_dim: Optional[int] = None,
        use_perceiver_frontend: bool = False,
        perceiver_latent_dim: int = 512,
        perceiver_num_latents: int = 8,
        perceiver_layers: int = 2,
        perceiver_heads: int = 8,
        perceiver_ff_mult: int = 4,
        enable_base_branch: bool = False,
    ):
        super().__init__()
        self.target_specs = list(target_specs)
        self.use_residual = use_residual
        self.residual_scale = residual_scale
        self.head_mode = head_mode
        self.fusion_mode = fusion_mode
        self.identity_dim = identity_dim
        self.use_perceiver_frontend = use_perceiver_frontend
        self.enable_base_branch = enable_base_branch
        if fusion_mode != "style_only" and identity_dim is None:
            raise ValueError("identity_dim is required when fusion_mode is not 'style_only'")

        self.fusion = EmbeddingFusion(style_dim=embed_dim, identity_dim=identity_dim, mode=fusion_mode)

        input_dim = perceiver_latent_dim if use_perceiver_frontend else self.fusion.output_dim
        self.residual_projector: Optional[nn.Linear] = None
        if use_residual:
            res_dim = residual_dim if residual_dim is not None else embed_dim
            self.residual_projector = ResidualProjector(res_dim, embed_dim, scale=residual_scale)
            input_dim += embed_dim

        self.norm = nn.LayerNorm(input_dim) if normalization else nn.Identity()
        self.trunk = None
        trunk_out_dim = input_dim
        if head_mode == "trunk":
            self.trunk = ResidualMLP(input_dim=input_dim, hidden_dim=trunk_hidden_dim, layers=trunk_layers)
            trunk_out_dim = self.trunk.out_dim
        self.heads = nn.ModuleDict(
            {spec.name: PolyLoRAHead(trunk_out_dim, spec, head_hidden_dim, enable_base_branch=enable_base_branch) for spec in self.target_specs}
        )
        self.perceiver: Optional[PerceiverMixer] = None
        if use_perceiver_frontend:
            self.perceiver = PerceiverMixer(
                input_dim=self.fusion.style_dim,
                latent_dim=perceiver_latent_dim,
                num_latents=perceiver_num_latents,
                num_layers=perceiver_layers,
                num_heads=perceiver_heads,
                ff_mult=perceiver_ff_mult,
            )
        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if ".proj." in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
        # Zero B branch and scale A/residual to stabilize, mirroring Qwen i2L init
        with torch.no_grad():
            for name, param in self.named_parameters():
                if ".lora_up" in name or name.endswith(".bias"):
                    continue
                if "proj." in name:
                    param.mul_(0.3)

    def forward(
        self,
        embedding: torch.Tensor,
        identity: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
        use_perceiver: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        use_perceiver = self.use_perceiver_frontend if use_perceiver is None else use_perceiver
        # Apply fusion (style + optional identity)
        fused = self.fusion(embedding, identity)
        if use_perceiver and self.perceiver is not None:
            tokens = [embedding]
            if identity is not None:
                id_tok = identity
                if self.fusion.identity_proj is not None:
                    id_tok = self.fusion.identity_proj(identity)
                tokens.append(id_tok)
            tokens_tensor = torch.stack(tokens, dim=1)
            fused = self.perceiver(tokens_tensor)
        hidden_in = fused
        if residual is not None and self.use_residual and self.residual_projector is not None:
            if residual.dim() == 1:
                residual = residual.unsqueeze(0)
            residual = self.residual_projector(residual)
            hidden_in = torch.cat([fused, residual], dim=-1)
        hidden = self.norm(hidden_in)
        trunk_out = self.trunk(hidden) if self.trunk is not None else hidden
        outputs: Dict[str, torch.Tensor] = {}
        for spec in self.target_specs:
            outputs.update(self.heads[spec.name](trunk_out))
        return outputs


def predict_lora_state_dict(
    model: PolyLoRANetwork,
    embedding: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    identity: Optional[torch.Tensor] = None,
    detach: bool = True,
    use_perceiver: Optional[bool] = None,
    include_base: bool = False,
) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        out = model(
            embedding,
            identity=identity,
            residual=residual,
            use_perceiver=use_perceiver,
        )
    filtered = {}
    for k, v in out.items():
        if not include_base and k.startswith("base."):
            continue
        filtered[k] = v.detach().cpu() if detach else v
    return filtered


def lora_loss(
    pred: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    base_target: Optional[Dict[str, torch.Tensor]] = None,
    cosine_weight: float = 0.0,
    base_weight: float = 1.0,
) -> torch.Tensor:
    """L1 + optional cosine similarity on LoRA A/B weights."""
    losses: List[torch.Tensor] = []
    for key, tgt in target.items():
        if key.endswith(".alpha") or key not in pred:
            continue
        losses.append(F.l1_loss(pred[key], tgt))
        if cosine_weight > 0:
            pred_flat = pred[key].reshape(pred[key].shape[0], -1)
            tgt_flat = tgt.reshape(tgt.shape[0], -1)
            losses.append(cosine_weight * (1 - F.cosine_similarity(pred_flat, tgt_flat, dim=-1)).mean())
    if base_target and base_weight > 0:
        for key, tgt in base_target.items():
            pred_key = f"base.{key}"
            if key.endswith(".alpha") or pred_key not in pred:
                continue
            losses.append(base_weight * F.l1_loss(pred[pred_key], tgt))
            if cosine_weight > 0:
                pred_flat = pred[pred_key].reshape(pred[pred_key].shape[0], -1)
                tgt_flat = tgt.reshape(tgt.shape[0], -1)
                losses.append(base_weight * cosine_weight * (1 - F.cosine_similarity(pred_flat, tgt_flat, dim=-1)).mean())
    if not losses:
        raise ValueError("No overlapping LoRA keys between pred and target.")
    return torch.stack(losses).mean()
